"""Quiet gate: background LLM work waits until no turn is running (local patch).

Not upstream. Inert unless ``KIROCREW_BG_QUIET_SECS`` is above zero AND the
gateway has called :func:`install`, so a CLI process, a test, or a gateway
started without the variable behaves exactly like stock Kiro Crew.

"Quiet" means: no session outside the background keys has a turn in flight and
the sub-agent manager reports no running sub-agent, observed continuously for
``KIROCREW_BG_QUIET_SECS`` seconds. Background keys are the gateway's own
sessions (``_bg``, ``_consolidate``, ``_hb`` and every other ``_``-prefixed key)
and per-store member consolidation sessions (``memory-consolidation:...``).

Three hooks call into this module, each a decorator or a subclass, so the
upstream files change by a few lines:

* ``consolidation_gate.consolidation_pass_gate()`` returns :class:`QuietGate`:
  a consolidation pass waits for quiet BEFORE it takes the one-slot gate, so it
  holds no lease and no transcript snapshot while it waits.
* ``history_consolidation_pass._call_llm`` is wrapped by
  :func:`preemptible_consolidation_call`: every consolidation model call
  (threshold, history, restart seeds, skill detection) runs one at a time in the
  background lane. When a foreground turn starts while it runs, the
  consolidation session's in-flight turn is cancelled (``session/cancel``; the
  harness aborts its HTTP request and the model server drops the request) and
  the SAME prompt runs again after the next quiet window. The caller never sees
  the interrupted attempt, so it is never charged to the retry budget.
* ``background_turns.run_bg_oneliner`` is wrapped by :func:`deferred_oneliner`:
  one-shot calls whose ``sel_source`` is listed in ``KIROCREW_BG_DEFER_SOURCES``
  wait for quiet in the same lane. They are short, so they are never
  interrupted; every other source (titles, link labels, dictation, the failure
  canary) and every FOREGROUND call runs at once, as in stock.

Settings (environment, read on each call):

* ``KIROCREW_BG_QUIET_SECS``: quiet window in seconds; ``0``/unset turns the gate off.
* ``KIROCREW_BG_POLL_SECS``: how often activity is sampled (default 2, min 0.05).
* ``KIROCREW_BG_MAX_DEFER_SECS``: longest a call waits before it runs anyway
  (default 0 = no limit). A call released this way is not interrupted.
* ``KIROCREW_BG_PREEMPT``: ``0`` keeps the gate but never interrupts a running call.
* ``KIROCREW_BG_DEFER_SOURCES``: comma-separated ``sel_source`` values to defer;
  replaces :data:`DEFAULT_DEFER_SOURCES` when set (an empty value defers none).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger("kiro_crew.bg_quiet")

T = TypeVar("T")

#: ``run_bg_oneliner`` sources deferred by default: automatic work nobody is
#: waiting on. Absent from the list (so never delayed): titles and older-session
#: summaries (default source), ``chat_nav`` link labels, ``stt_polish``,
#: ``stt_endpointing``, ``poisoned_canary``, ``slack_code_channel_router``,
#: ``wake_judge``.
DEFAULT_DEFER_SOURCES = frozenset(
    {
        "contradiction_check",
        "tips",
        "suggestions",
        "chat_folders",
        "chat_folder_suggest",
        "chat_auto_tag",
        "session_summary",
        "dynamic_dashboard_card",
        "warm_task_brief",
    }
)

#: Session-key prefixes that are background work, never "a chat is busy".
BACKGROUND_KEY_PREFIXES = ("_", "memory-consolidation:")

#: Keys whose in-flight turn :func:`preemptible_consolidation_call` may cancel.
_CONSOLIDATE_KEY = "_consolidate"
_MEMBER_CONSOLIDATION_PREFIX = "memory-consolidation:"

#: A gap between two observations longer than this many poll intervals breaks
#: the observed-quiet chain: nobody was looking, so the window restarts.
_CHAIN_GAP_POLLS = 3.0


@dataclass
class _State:
    sessions: Any = None
    subagents: Any = None
    quiet_since: float | None = None
    last_observed: float | None = None


_STATE = _State()
_LANE: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def install(sessions: Any, subagents: Any) -> None:
    """Register the gateway's session manager and sub-agent manager."""
    _STATE.sessions = sessions
    _STATE.subagents = subagents
    _STATE.quiet_since = None
    _STATE.last_observed = None
    if quiet_secs() > 0:
        logger.info(
            "bg_quiet: background LLM work waits for %.0fs without running turns "
            "(poll %.1fs, max defer %s, preempt %s)",
            quiet_secs(),
            poll_secs(),
            f"{max_defer_secs():.0f}s" if max_defer_secs() > 0 else "none",
            "on" if preempt_enabled() else "off",
        )


def uninstall() -> None:
    """Forget the registered managers (tests)."""
    global _LANE
    _STATE.sessions = None
    _STATE.subagents = None
    _STATE.quiet_since = None
    _STATE.last_observed = None
    _LANE = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value == value and value >= 0 else default  # NaN/negative -> default


def quiet_secs() -> float:
    return _env_float("KIROCREW_BG_QUIET_SECS", 0.0)


def poll_secs() -> float:
    return max(0.05, _env_float("KIROCREW_BG_POLL_SECS", 2.0))


def max_defer_secs() -> float:
    return _env_float("KIROCREW_BG_MAX_DEFER_SECS", 0.0)


def preempt_enabled() -> bool:
    return os.environ.get("KIROCREW_BG_PREEMPT", "1").strip() not in ("0", "false", "no", "off")


def defer_sources() -> frozenset[str]:
    raw = os.environ.get("KIROCREW_BG_DEFER_SOURCES")
    if raw is None:
        return DEFAULT_DEFER_SOURCES
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def enabled() -> bool:
    """True when the gate is configured AND the gateway registered its managers."""
    return _STATE.sessions is not None and quiet_secs() > 0


def _is_background_key(key: object) -> bool:
    return isinstance(key, str) and key.startswith(BACKGROUND_KEY_PREFIXES)


def _has_active_turn(provider: Any) -> bool:
    fn = getattr(provider, "has_active_turn", None)
    if not callable(fn):
        return False
    try:
        return fn() is True
    except Exception:
        return False


def _session_items() -> list[tuple[Any, Any]]:
    table = getattr(_STATE.sessions, "_sessions", None)
    if table is None:
        return []
    try:
        return list(table.items())
    except Exception:
        logger.debug("bg_quiet: session table unreadable", exc_info=True)
        return []


def foreground_busy() -> bool:
    """True when a non-background session has a turn in flight or a sub-agent runs."""
    for key, entry in _session_items():
        if _is_background_key(key):
            continue
        if _has_active_turn(getattr(entry, "provider", None)):
            return True
    try:
        running = int(getattr(_STATE.subagents, "running_count", 0) or 0)
    except Exception:
        running = 0
    return running > 0


def _observe(now: float | None = None) -> bool:
    """Sample activity once, extend or reset the observed-quiet chain, return busy."""
    now = time.monotonic() if now is None else now
    busy = foreground_busy()
    gap = None if _STATE.last_observed is None else now - _STATE.last_observed
    if busy:
        _STATE.quiet_since = None
    elif _STATE.quiet_since is None or gap is None or gap > _CHAIN_GAP_POLLS * poll_secs():
        _STATE.quiet_since = now
    _STATE.last_observed = now
    return busy


def is_quiet(now: float | None = None) -> bool:
    """Sample now; True when quiet has been observed for the whole window."""
    now = time.monotonic() if now is None else now
    if _observe(now):
        return False
    since = _STATE.quiet_since
    return since is not None and now - since >= quiet_secs()


async def wait_until_quiet(label: str = "background call") -> bool:
    """Wait for the quiet window. True when reached (or the gate is off), False on max defer."""
    if not enabled():
        return True
    started = time.monotonic()
    announced = False
    while True:
        now = time.monotonic()
        if is_quiet(now):
            if announced:
                logger.info("bg_quiet: %s starts after waiting %.0fs", label, now - started)
            return True
        limit = max_defer_secs()
        if limit > 0 and now - started >= limit:
            logger.warning(
                "bg_quiet: %s waited %.0fs (KIROCREW_BG_MAX_DEFER_SECS); running anyway",
                label,
                now - started,
            )
            return False
        if not announced:
            logger.info("bg_quiet: %s deferred until the instance is quiet", label)
            announced = True
        await asyncio.sleep(poll_secs())


def _lane() -> asyncio.Lock:
    """The running loop's one-at-a-time lane for deferred background calls (FIFO)."""
    global _LANE
    loop = asyncio.get_running_loop()
    if _LANE is None or _LANE[0] is not loop:
        _LANE = (loop, asyncio.Lock())
    return _LANE[1]


class QuietGate(asyncio.Semaphore):
    """``asyncio.Semaphore`` whose ``acquire`` first waits for quiet when the gate is on."""

    async def acquire(self) -> bool:  # type: ignore[override]
        await wait_until_quiet("memory consolidation pass")
        return await super().acquire()


def _should_defer(kwargs: dict[str, Any]) -> bool:
    priority = kwargs.get("start_priority")
    if getattr(priority, "value", priority) == "fg":
        return False
    return str(kwargs.get("sel_source", "bg_oneliner")) in defer_sources()


def deferred_oneliner(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Wrap ``run_bg_oneliner``: listed sources wait for quiet in the lane, others pass through."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        if not enabled() or not _should_defer(kwargs):
            return await fn(*args, **kwargs)
        source = str(kwargs.get("sel_source", "bg_oneliner"))
        async with _lane():
            await wait_until_quiet(f"background call ({source})")
            return await fn(*args, **kwargs)

    return wrapper


async def _cancel_consolidation_turns() -> int:
    """Cancel every consolidation session's in-flight turn; return how many were sent."""
    sent = 0
    for key, entry in _session_items():
        if not isinstance(key, str):
            continue
        if key != _CONSOLIDATE_KEY and not key.startswith(_MEMBER_CONSOLIDATION_PREFIX):
            continue
        provider = getattr(entry, "provider", None)
        if not _has_active_turn(provider):
            continue
        cancel = getattr(provider, "cancel", None)
        if not callable(cancel):
            continue
        try:
            await cancel()
            sent += 1
        except Exception:
            logger.debug("bg_quiet: cancel of %s failed", key, exc_info=True)
    return sent


async def _watch_for_foreground(preempted: asyncio.Event) -> None:
    """While a consolidation call runs, cancel its turn whenever a foreground turn appears."""
    while True:
        await asyncio.sleep(poll_secs())
        if _observe() and await _cancel_consolidation_turns():
            if not preempted.is_set():
                logger.info("bg_quiet: a turn started; interrupting the consolidation model call")
            preempted.set()


def preemptible_consolidation_call(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Wrap ``HistoryConsolidator._call_llm``: wait for quiet, run, re-run if interrupted."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        if not enabled():
            return await fn(*args, **kwargs)
        async with _lane():
            attempt = 0
            while True:
                attempt += 1
                reached = await wait_until_quiet("consolidation model call")
                if not reached or not preempt_enabled():
                    return await fn(*args, **kwargs)
                preempted = asyncio.Event()
                watcher = asyncio.ensure_future(_watch_for_foreground(preempted))
                try:
                    result = await fn(*args, **kwargs)
                except Exception:
                    if not preempted.is_set():
                        raise
                    result = None
                finally:
                    watcher.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await watcher
                if not preempted.is_set():
                    return result  # type: ignore[return-value]
                logger.info(
                    "bg_quiet: consolidation model call interrupted (attempt %d); "
                    "re-running after the next quiet window",
                    attempt,
                )

    return wrapper


#: Short names the hooks use. The hooks reach them through ``__import__`` so the
#: owner modules bind no new name: their facades serve every name an owner binds,
#: and the facade-shape tests refuse a name the original module did not have.
defer = deferred_oneliner
preemptible = preemptible_consolidation_call
