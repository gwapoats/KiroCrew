"""Quiet gate (local patch): background LLM work waits for an instance with no running turn."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable

import pytest

from kiro_crew import bg_quiet
from kiro_crew.start_priority import StartPriority

QUIET = 0.2
POLL = 0.02


class _Provider:
    def __init__(self) -> None:
        self.active = False
        self.cancels = 0
        self.cancelled = asyncio.Event()

    def has_active_turn(self) -> bool:
        return self.active

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> str:
        self.cancels += 1
        self.active = False
        self.cancelled.set()
        return "acked"


class _Entry:
    def __init__(self, provider: _Provider) -> None:
        self.provider = provider


class _Sessions:
    def __init__(self) -> None:
        self._sessions: dict[str, _Entry] = {}

    def add(self, key: str) -> _Provider:
        provider = _Provider()
        self._sessions[key] = _Entry(provider)
        return provider


class _Subagents:
    running_count = 0


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "KIROCREW_BG_QUIET_SECS",
        "KIROCREW_BG_POLL_SECS",
        "KIROCREW_BG_MAX_DEFER_SECS",
        "KIROCREW_BG_PREEMPT",
        "KIROCREW_BG_DEFER_SOURCES",
    ):
        monkeypatch.delenv(name, raising=False)
    bg_quiet.uninstall()
    yield
    bg_quiet.uninstall()


def _enable(monkeypatch: pytest.MonkeyPatch, **extra: str) -> tuple[_Sessions, _Subagents]:
    monkeypatch.setenv("KIROCREW_BG_QUIET_SECS", str(QUIET))
    monkeypatch.setenv("KIROCREW_BG_POLL_SECS", str(POLL))
    for name, value in extra.items():
        monkeypatch.setenv(name, value)
    sessions, subagents = _Sessions(), _Subagents()
    bg_quiet.install(sessions, subagents)
    return sessions, subagents


async def _until(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _recording_oneliner() -> tuple[Callable[..., Any], list[str]]:
    calls: list[str] = []

    async def oneliner(sessions: Any, prompt: str, **kwargs: Any) -> str:
        calls.append(prompt)
        return f"ok:{prompt}"

    return bg_quiet.deferred_oneliner(oneliner), calls


@pytest.mark.asyncio
async def test_without_env_the_wrappers_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _Sessions(), _Subagents()
    sessions.add("dashboard:chat-1").active = True
    bg_quiet.install(sessions, _Subagents())
    wrapped, calls = _recording_oneliner()
    assert not bg_quiet.enabled()
    assert await asyncio.wait_for(wrapped(None, "a", sel_source="tips"), 1) == "ok:a"
    assert calls == ["a"]


@pytest.mark.asyncio
async def test_not_installed_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIROCREW_BG_QUIET_SECS", "60")
    wrapped, calls = _recording_oneliner()
    assert not bg_quiet.enabled()
    assert await asyncio.wait_for(wrapped(None, "a", sel_source="tips"), 1) == "ok:a"


@pytest.mark.asyncio
async def test_deferred_source_waits_until_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch)
    chat = sessions.add("dashboard:chat-1")
    chat.active = True
    wrapped, calls = _recording_oneliner()
    task = asyncio.ensure_future(wrapped(None, "check", sel_source="contradiction_check"))
    await asyncio.sleep(QUIET * 2)
    assert not task.done() and calls == []
    chat.active = False
    started = time.monotonic()
    assert await asyncio.wait_for(task, 5) == "ok:check"
    assert time.monotonic() - started >= QUIET * 0.9


@pytest.mark.asyncio
async def test_undeferred_sources_and_foreground_run_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, _ = _enable(monkeypatch)
    sessions.add("dashboard:chat-1").active = True
    wrapped, calls = _recording_oneliner()
    assert await asyncio.wait_for(wrapped(None, "title"), 1) == "ok:title"
    assert await asyncio.wait_for(wrapped(None, "stt", sel_source="stt_polish"), 1) == "ok:stt"
    fg = wrapped(
        None, "fg", sel_source="contradiction_check", start_priority=StartPriority.FOREGROUND
    )
    assert await asyncio.wait_for(fg, 1) == "ok:fg"
    assert calls == ["title", "stt", "fg"]


@pytest.mark.asyncio
async def test_defer_sources_env_replaces_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch, KIROCREW_BG_DEFER_SOURCES="bg_oneliner")
    chat = sessions.add("dashboard:chat-1")
    chat.active = True
    wrapped, _calls = _recording_oneliner()
    assert await asyncio.wait_for(wrapped(None, "tips", sel_source="tips"), 1) == "ok:tips"
    title = asyncio.ensure_future(wrapped(None, "title"))
    await asyncio.sleep(QUIET * 2)
    assert not title.done()
    chat.active = False
    assert await asyncio.wait_for(title, 5) == "ok:title"


@pytest.mark.asyncio
async def test_background_keys_never_count_as_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch)
    for key in ("_bg", "_consolidate", "_hb", "memory-consolidation:store:abc"):
        sessions.add(key).active = True
    assert not bg_quiet.foreground_busy()
    wrapped, _calls = _recording_oneliner()
    assert await asyncio.wait_for(wrapped(None, "t", sel_source="tips"), 5) == "ok:t"


@pytest.mark.asyncio
async def test_running_subagent_counts_as_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    _sessions, subagents = _enable(monkeypatch)
    subagents.running_count = 1
    assert bg_quiet.foreground_busy()
    wrapped, calls = _recording_oneliner()
    task = asyncio.ensure_future(wrapped(None, "t", sel_source="tips"))
    await asyncio.sleep(QUIET * 2)
    assert not task.done()
    subagents.running_count = 0
    assert await asyncio.wait_for(task, 5) == "ok:t"


@pytest.mark.asyncio
async def test_deferred_calls_run_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    running = 0
    peak = 0

    async def slow(sessions: Any, prompt: str, **kwargs: Any) -> str:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return prompt

    wrapped = bg_quiet.deferred_oneliner(slow)
    results = await asyncio.wait_for(
        asyncio.gather(*(wrapped(None, str(i), sel_source="tips") for i in range(4))), 10
    )
    assert results == ["0", "1", "2", "3"]
    assert peak == 1


@pytest.mark.asyncio
async def test_max_defer_runs_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch, KIROCREW_BG_MAX_DEFER_SECS="0.3")
    sessions.add("dashboard:chat-1").active = True
    wrapped, calls = _recording_oneliner()
    started = time.monotonic()
    assert await asyncio.wait_for(wrapped(None, "t", sel_source="tips"), 5) == "ok:t"
    assert time.monotonic() - started >= 0.25


def _fake_call_llm(consolidate: _Provider, plan: list[str]) -> tuple[Callable[..., Any], list[int]]:
    """A ``_call_llm`` stand-in: each attempt pops a behaviour from *plan*."""
    attempts: list[int] = []

    async def call_llm(self: Any, prompt: str, *, session_key: str = "") -> Any:
        attempts.append(len(attempts) + 1)
        behaviour = plan.pop(0)
        consolidate.active = True
        consolidate.cancelled.clear()
        if behaviour == "finish":
            consolidate.active = False
            return {"prompt": prompt, "attempt": len(attempts)}
        await asyncio.wait_for(consolidate.cancelled.wait(), 5)
        if behaviour == "raise-on-cancel":
            raise RuntimeError("turn ended by cancel")
        return None

    return bg_quiet.preemptible_consolidation_call(call_llm), attempts


@pytest.mark.asyncio
async def test_consolidation_call_is_interrupted_and_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch)
    consolidate = sessions.add("_consolidate")
    chat = sessions.add("dashboard:chat-1")
    wrapped, attempts = _fake_call_llm(consolidate, ["block", "finish"])
    task = asyncio.ensure_future(wrapped(object(), "P", session_key="dashboard:chat-2"))
    await _until(lambda: consolidate.active)
    chat.active = True
    await _until(lambda: consolidate.cancels >= 1)
    await asyncio.sleep(QUIET)
    assert not task.done()
    chat.active = False
    result = await asyncio.wait_for(task, 5)
    assert result == {"prompt": "P", "attempt": 2}
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_an_error_caused_by_the_interrupt_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, _ = _enable(monkeypatch)
    consolidate = sessions.add("_consolidate")
    chat = sessions.add("dashboard:chat-1")
    wrapped, attempts = _fake_call_llm(consolidate, ["raise-on-cancel", "finish"])
    task = asyncio.ensure_future(wrapped(object(), "P"))
    await _until(lambda: consolidate.active)
    chat.active = True
    await _until(lambda: consolidate.cancels >= 1)
    chat.active = False
    assert (await asyncio.wait_for(task, 5))["attempt"] == 2


@pytest.mark.asyncio
async def test_an_error_without_interrupt_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)

    async def failing(self: Any, prompt: str) -> Any:
        raise ValueError("model said no")

    wrapped = bg_quiet.preemptible_consolidation_call(failing)
    with pytest.raises(ValueError):
        await asyncio.wait_for(wrapped(object(), "P"), 5)


@pytest.mark.asyncio
async def test_member_consolidation_turns_are_interrupted_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, _ = _enable(monkeypatch)
    member = sessions.add("memory-consolidation:store:1")
    chat = sessions.add("dashboard:chat-1")
    wrapped, attempts = _fake_call_llm(member, ["block", "finish"])
    task = asyncio.ensure_future(wrapped(object(), "P"))
    await _until(lambda: member.active)
    chat.active = True
    await _until(lambda: member.cancels >= 1)
    chat.active = False
    assert (await asyncio.wait_for(task, 5))["attempt"] == 2


@pytest.mark.asyncio
async def test_preempt_off_lets_the_call_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions, _ = _enable(monkeypatch, KIROCREW_BG_PREEMPT="0")
    consolidate = sessions.add("_consolidate")
    chat = sessions.add("dashboard:chat-1")
    finish = asyncio.Event()

    async def call_llm(self: Any, prompt: str) -> str:
        consolidate.active = True
        await finish.wait()
        consolidate.active = False
        return "done"

    task = asyncio.ensure_future(bg_quiet.preemptible_consolidation_call(call_llm)(object(), "P"))
    await _until(lambda: consolidate.active)
    chat.active = True
    await asyncio.sleep(POLL * 10)
    assert consolidate.cancels == 0
    finish.set()
    assert await asyncio.wait_for(task, 5) == "done"


@pytest.mark.asyncio
async def test_consolidation_gate_waits_for_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.12+: the one-slot pass gate. 0.11 has none; its hook is checked below.
    consolidation_gate = pytest.importorskip("kiro_crew.consolidation_gate")

    monkeypatch.setattr(consolidation_gate, "_GATE", None)
    gate = consolidation_gate.consolidation_pass_gate()
    assert isinstance(gate, bg_quiet.QuietGate)
    assert isinstance(gate, asyncio.Semaphore)
    # Off: acquires at once, as stock.
    await asyncio.wait_for(gate.acquire(), 1)
    gate.release()
    sessions, _ = _enable(monkeypatch)
    chat = sessions.add("dashboard:chat-1")
    chat.active = True
    waiter = asyncio.ensure_future(gate.acquire())
    await asyncio.sleep(QUIET * 2)
    assert not waiter.done()
    assert not gate.locked()
    chat.active = False
    await asyncio.wait_for(waiter, 5)
    assert gate.locked()
    gate.release()


def test_upstream_hooks_are_in_place() -> None:
    import importlib.util

    from kiro_crew import history_consolidation_pass
    from kiro_crew.history import HistoryConsolidator
    from kiro_crew.llm_helpers import run_bg_oneliner
    from kiro_crew.slack.gateway_runtime import subagent_wiring

    assert inspect.unwrap(run_bg_oneliner) is not run_bg_oneliner
    assert "start_priority" in inspect.signature(run_bg_oneliner).parameters
    assert inspect.iscoroutinefunction(run_bg_oneliner)
    call_llm = HistoryConsolidator._call_llm
    assert inspect.unwrap(call_llm) is not call_llm
    assert "bg_quiet.install(self.sessions, self.subagent_mgr)" in inspect.getsource(
        subagent_wiring._init_subagents
    )
    if importlib.util.find_spec("kiro_crew.consolidation_gate") is None:
        # 0.11: the pass waits for quiet before it takes its lease.
        source = inspect.getsource(history_consolidation_pass._consolidate)
        assert "wait_until_quiet(" in source
        assert source.index("wait_until_quiet(") < source.index("_acquire_consolidation_lease")
