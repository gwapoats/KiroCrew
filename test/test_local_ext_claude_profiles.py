"""LOCAL PATCH: ``kiro_crew.local_ext.claude_profiles`` -- a chat on a second Claude config.

Unit tests of the module's decisions and of how ``chat_backends`` carries them;
real runs of a dashboard turn and of the prewarm, which is where the call sites in
upstream files are checked (by what they do, not by how they are spelled); and
"seam canaries" that pin the upstream names the patch leans on, so a rebase onto a
release that moved one fails here first rather than silently running a profiled
chat on ``~/.claude``.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from eager_spawn_helpers import _cfg as _eager_cfg
from eager_spawn_helpers import (  # noqa: F401 -- fixtures are used by name
    _isolate_armed_registry,
    _mock_state,
    _pin_prewarm_allowance,
)
from turn_harness import ScriptedProvider, TurnScript, run_turn

from kiro_crew.agent_sdk import backends as _backends
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.config.loader import ResolvedBindings
from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable
from kiro_crew.dashboard import chat_runner
from kiro_crew.dashboard.state import _ChatSlot
from kiro_crew.execution_context import ExecutionContext, MemoryStoreRef
from kiro_crew.local_ext import chat_backends as cb
from kiro_crew.local_ext import claude_profile_dirs as dirs
from kiro_crew.local_ext import claude_profiles as cp

KEY = "dashboard:chat-1"
SID = "0b6f1c1e-3a52-4c8d-9a55-6f0d2c1e7a10"
#: The template the turn harness's default slot runs.
TEMPLATE = "kirocrew"


def _cfg(
    profiles=None,
    *,
    routes=None,
    profiles_on=True,
    routes_on=False,
    default="claude",
    member="kas",
    agent_env=None,
):
    local_ext = {}
    if profiles_on:
        local_ext["claude_profiles"] = {} if profiles is None else {"agents": profiles}
    if routes_on or routes is not None:
        local_ext["chat_backends"] = {} if routes is None else {"agents": routes}
    return SimpleNamespace(
        _extra_sections={"local_ext": local_ext} if local_ext else {},
        agent=SimpleNamespace(
            acp_backend=default, member_acp_backend=member, kas_bundle=False, env=agent_env or {}
        ),
        agents={},
    )


def _slot(**over):
    base = dict(
        key="chat-1",
        agent="anthropic",
        agent_kind="template",
        project="",
        mode="",
        _app="",
        app_agent_owner="",
        linked_session_key="",
        _slack_channel="",
        _slack_linked=False,
        channel_origin=False,
        _channel_runtime_origin=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _template_exec(name="anthropic"):
    return SimpleNamespace(selection_kind="template", member_id=None, template_id=name)


class _Sessions:
    def __init__(self, provider=None, *, override=None, env=None):
        self._provider = provider
        self._sessions = {KEY: SimpleNamespace(backend_override=override, extra_env=env)}

    def _fold_key(self, key):
        return key

    def get_provider(self, key):
        return self._provider

    def get_mirror_link(self, key):
        return None


def _provider(backend):
    return SimpleNamespace(capabilities=capabilities_for(backend))


@pytest.fixture(autouse=True)
def _selectable(monkeypatch):
    """A deterministic selectable set, whatever this host registered."""
    monkeypatch.setattr(_backends, "_selectable", {"", "kas", "claude", "codex"})
    monkeypatch.setattr(_backends, "_host_unservable", {})


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """No provider variable inherited from whatever started the test run."""
    for name in cp.PROVIDER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home directory of the test's own, with no profile directory in it yet."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return tmp_path


@pytest.fixture
def signed_in(home):
    """The ``cloud`` profile's directory exists, as it does after a terminal sign-in."""
    directory = home / ".claude-cloud"
    directory.mkdir()
    return directory


@pytest.fixture
def armed(monkeypatch):
    calls = []
    from kiro_crew.dashboard import session_control

    monkeypatch.setattr(session_control, "arm_model_switch_reset", lambda st, sl: calls.append(sl))
    return calls


# ── the directory table ──────────────────────────────────────────────────────


class TestProfileDirs:
    def test_cloud_is_a_sibling_of_the_default_config(self, home):
        assert dirs.config_dir("cloud") == os.path.join(str(home), ".claude-cloud")
        assert not dirs.config_dir("cloud").endswith(os.sep)

    def test_unknown_profile_has_no_directory(self):
        with pytest.raises(KeyError):
            dirs.config_dir("nope")

    def test_leaf_tables_follow_the_directory_table(self):
        for directory in dirs.PROFILE_DIRS.values():
            assert f"{directory}/.credentials.json" in dirs.CREDENTIAL_LEAVES
            assert f"{directory}/settings.json" in dirs.SETTINGS_LEAVES
            assert f"{directory}/settings.local.json" in dirs.SETTINGS_LEAVES

    def test_existing_config_dirs_lists_only_what_is_on_disk(self, home):
        assert dirs.existing_config_dirs() == ()
        (home / ".claude-cloud").mkdir()
        assert dirs.existing_config_dirs() == (str(home / ".claude-cloud"),)


# ── configuration ────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_absent_section_is_off(self):
        cfg = _cfg(profiles_on=False)
        assert cp.feature_section(cfg) is None
        assert cp.agent_profiles(cfg) is None
        assert cp.profile_for_template(cfg, "anthropic") is None

    @pytest.mark.parametrize("bad", ["cloud", ["cloud"], 3, None])
    def test_non_object_section_is_off(self, bad):
        cfg = _cfg(profiles_on=False)
        cfg._extra_sections = {"local_ext": {"claude_profiles": bad}}
        assert cp.agent_profiles(cfg) is None

    def test_configured_without_map_is_empty(self):
        assert cp.agent_profiles(_cfg(None)) == {}

    def test_lookup(self):
        cfg = _cfg({"anthropic": "cloud", "spaced": " cloud ", "plain": ""})
        assert cp.profile_for_template(cfg, "anthropic") == "cloud"
        assert cp.profile_for_template(cfg, "spaced") == "cloud"
        assert cp.profile_for_template(cfg, "plain") is None
        assert cp.profile_for_template(cfg, "other") is None
        assert cp.profile_for_template(cfg, None) is None

    @pytest.mark.parametrize("stored", ["work", "sk-secret-token", "/tmp/elsewhere", 7, ["cloud"]])
    def test_unknown_profile_fails_closed_without_echoing(self, stored):
        with pytest.raises(cp.ClaudeProfileUnavailable) as info:
            cp.profile_for_template(_cfg({"anthropic": stored}), "anthropic")
        exc = info.value
        assert isinstance(exc, CrewmateHarnessUnavailable)
        assert exc.code == "claude_profile_unavailable"
        assert exc.stored == "claude" and exc.crew == ""
        assert "anthropic" in str(exc) and "cloud" in str(exc)
        assert "sk-secret-token" not in str(exc) and "/tmp/elsewhere" not in str(exc)

    def test_profile_env_names_only_the_config_directory(self, home):
        assert cp.profile_env("cloud") == {"CLAUDE_CONFIG_DIR": str(home / ".claude-cloud")}

    def test_section_survives_load_and_save(self, tmp_path, monkeypatch, caplog):
        import json
        import logging

        from kiro_crew.config import loader as L

        section = {"claude_profiles": {"agents": {"anthropic": "cloud"}}}
        cfgp = tmp_path / "config.json"
        cfgp.write_text(json.dumps({"agent": {"provider": "acp"}, "local_ext": section}))
        monkeypatch.setattr(L, "config_path", lambda: cfgp)
        monkeypatch.setattr(L, "config_dir", lambda: tmp_path)
        monkeypatch.setattr(L, "config_local_path", lambda: tmp_path / "config.local.json")
        with caplog.at_level(logging.WARNING):
            cfg = L.KiroCrewConfig.load()
        assert not [r for r in caplog.records if "unrecognized top-level keys" in r.getMessage()]
        assert cfg.to_dict()["local_ext"] == section
        assert cp.agent_profiles(cfg) == {"anthropic": "cloud"}


# ── whether a profile can start ──────────────────────────────────────────────


class TestEnsureStartable:
    @pytest.mark.asyncio
    async def test_signed_in_profile_on_claude_starts(self, signed_in):
        await cp.ensure_startable(_cfg({"anthropic": "cloud"}), "anthropic", "cloud", "claude")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["codex", "kas", ""])
    async def test_another_harness_is_refused(self, signed_in, backend):
        with pytest.raises(cp.ClaudeProfileUnavailable) as info:
            await cp.ensure_startable(_cfg(), "anthropic", "cloud", backend)
        assert "would not run on Claude Code" in str(info.value)
        assert "local_ext.chat_backends" in str(info.value)

    @pytest.mark.asyncio
    async def test_missing_directory_names_the_sign_in(self, home):
        with pytest.raises(cp.ClaudeProfileUnavailable) as info:
            await cp.ensure_startable(_cfg(), "anthropic", "cloud", "claude")
        assert 'CLAUDE_CONFIG_DIR="$HOME/.claude-cloud" claude' in str(info.value)
        assert str(home) not in str(info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", cp.PROVIDER_ENV_NAMES)
    async def test_inherited_provider_variable_is_refused_by_name(
        self, signed_in, monkeypatch, name
    ):
        monkeypatch.setenv(name, "value-that-must-not-be-echoed")
        with pytest.raises(cp.ClaudeProfileUnavailable) as info:
            await cp.ensure_startable(_cfg(), "anthropic", "cloud", "claude")
        assert name in str(info.value)
        assert "value-that-must-not-be-echoed" not in str(info.value)

    @pytest.mark.asyncio
    async def test_agent_env_provider_variable_is_refused(self, signed_in):
        cfg = _cfg(agent_env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:8000"})
        with pytest.raises(cp.ClaudeProfileUnavailable) as info:
            await cp.ensure_startable(cfg, "anthropic", "cloud", "claude")
        assert "ANTHROPIC_BASE_URL" in str(info.value)
        assert "127.0.0.1" not in str(info.value)

    def test_inherited_names_are_sorted_and_empty_values_ignored(self):
        env = {"ANTHROPIC_BASE_URL": "x", "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_USE_BEDROCK": "1"}
        assert cp.inherited_provider_env(_cfg(), env) == (
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
        )
        assert cp.inherited_provider_env(SimpleNamespace(), {}) == ()


# ── the turn ─────────────────────────────────────────────────────────────────


class TestTurnRoute:
    @pytest.mark.asyncio
    async def test_profile_rides_the_allocation_only(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions())
        got = await cb.turn_route(
            state, _slot(), KEY, _cfg({"anthropic": "cloud"}), "anthropic", _template_exec()
        )
        # The model ladder is spread the mapping itself and takes no ``extra_env``.
        assert got == {}
        assert got.allocation == {"extra_env": {"CLAUDE_CONFIG_DIR": str(signed_in)}}
        assert armed == []

    @pytest.mark.asyncio
    async def test_profile_and_route_travel_together(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions())
        cfg = _cfg({"anthropic": "cloud"}, routes={"anthropic": "claude"}, default="kas")
        got = await cb.turn_route(state, _slot(), KEY, cfg, "anthropic", _template_exec())
        assert got == {"backend_override": "claude"}
        assert got.allocation == {
            "backend_override": "claude",
            "extra_env": {"CLAUDE_CONFIG_DIR": str(signed_in)},
        }

    @pytest.mark.asyncio
    async def test_profile_on_another_harness_raises_before_any_claim(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        cfg = _cfg({"anthropic": "cloud"}, routes={"anthropic": "codex"})
        with pytest.raises(cp.ClaudeProfileUnavailable):
            await cb.turn_route(state, _slot(), KEY, cfg, "anthropic", _template_exec())
        assert armed == []

    @pytest.mark.asyncio
    async def test_unknown_profile_raises_before_any_claim(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        with pytest.raises(cp.ClaudeProfileUnavailable):
            await cb.turn_route(
                state, _slot(), KEY, _cfg({"anthropic": "work"}), "anthropic", _template_exec()
            )
        assert armed == []

    @pytest.mark.asyncio
    async def test_unsigned_profile_raises_before_any_claim(self, home, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        with pytest.raises(cp.ClaudeProfileUnavailable):
            await cb.turn_route(
                state, _slot(), KEY, _cfg({"anthropic": "cloud"}), "anthropic", _template_exec()
            )
        assert armed == []

    @pytest.mark.asyncio
    async def test_live_default_session_is_reset_onto_the_profile(self, signed_in, armed):
        slot = _slot()
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        got = await cb.turn_route(
            state, slot, KEY, _cfg({"anthropic": "cloud"}), "anthropic", _template_exec()
        )
        assert got.allocation["extra_env"] == {"CLAUDE_CONFIG_DIR": str(signed_in)}
        assert armed == [slot]

    @pytest.mark.asyncio
    async def test_live_session_already_under_the_profile_is_kept(self, signed_in, armed):
        env = {"CLAUDE_CONFIG_DIR": str(signed_in)}
        state = SimpleNamespace(sessions=_Sessions(_provider("claude"), env=env))
        await cb.turn_route(
            state, _slot(), KEY, _cfg({"anthropic": "cloud"}), "anthropic", _template_exec()
        )
        assert armed == []

    @pytest.mark.asyncio
    async def test_profile_removed_moves_our_session_back(self, signed_in, armed):
        slot = _slot()
        env = {"CLAUDE_CONFIG_DIR": str(signed_in)}
        state = SimpleNamespace(sessions=_Sessions(_provider("claude"), env=env))
        got = await cb.turn_route(state, slot, KEY, _cfg({}), "anthropic", _template_exec())
        assert got == {} and got.allocation == {}
        assert armed == [slot]

    @pytest.mark.asyncio
    async def test_unprofiled_chat_on_the_default_config_is_left_alone(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        got = await cb.turn_route(
            state,
            _slot(agent="kirocrew"),
            KEY,
            _cfg({"anthropic": "cloud"}),
            "kirocrew",
            _template_exec("kirocrew"),
        )
        assert got == {} and got.allocation == {}
        assert armed == []

    @pytest.mark.asyncio
    async def test_profiles_off_never_compares_config_directories(self, signed_in, armed):
        # Routes only: a stamp some other caller left is not this patch's to judge.
        env = {"CLAUDE_CONFIG_DIR": str(signed_in)}
        state = SimpleNamespace(sessions=_Sessions(_provider("claude"), env=env))
        cfg = _cfg(profiles_on=False, routes={})
        got = await cb.turn_route(state, _slot(), KEY, cfg, "anthropic", _template_exec())
        assert got == {} and got.allocation == {}
        assert armed == []

    @pytest.mark.asyncio
    async def test_nothing_configured_changes_nothing(self, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        got = await cb.turn_route(
            state, _slot(), KEY, _cfg(profiles_on=False), "anthropic", _template_exec()
        )
        assert got == {} and got.allocation == {}
        assert armed == []

    @pytest.mark.asyncio
    async def test_member_selection_takes_no_profile(self, signed_in, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        member = SimpleNamespace(selection_kind="member", member_id="m1")
        got = await cb.turn_route(
            state, _slot(), KEY, _cfg({"anthropic": "cloud"}), "anthropic", member
        )
        assert got == {} and got.allocation == {}
        assert armed == []

    def test_stamped_config_dir_reads_the_session_record(self):
        sessions = _Sessions(env={"CLAUDE_CONFIG_DIR": "/h/.claude-cloud", "AWS_PROFILE": "p"})
        assert cp.stamped_config_dir(sessions, KEY) == "/h/.claude-cloud"
        assert cp.stamped_config_dir(_Sessions(env={"AWS_PROFILE": "p"}), KEY) is None
        assert cp.stamped_config_dir(_Sessions(), KEY) is None
        assert cp.stamped_config_dir(SimpleNamespace(), KEY) is None


class TestPrewarm:
    def _bindings(self, agent="anthropic"):
        return SimpleNamespace(execution_context=None, selection_kind="template", kiro_agent=agent)

    @pytest.mark.asyncio
    async def test_profiled_chat_is_not_prewarmed(self):
        assert await cb.blocks_prewarm(
            None, _slot(), KEY, _cfg({"anthropic": "cloud"}), self._bindings()
        )

    @pytest.mark.asyncio
    async def test_broken_profile_is_not_prewarmed(self):
        assert await cb.blocks_prewarm(
            None, _slot(), KEY, _cfg({"anthropic": "work"}), self._bindings()
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cfg",
        [_cfg(profiles_on=False), _cfg({}), _cfg({"other": "cloud"}), _cfg({"anthropic": ""})],
    )
    async def test_everything_else_prewarms_as_before(self, cfg):
        assert not await cb.blocks_prewarm(None, _slot(), KEY, cfg, self._bindings())

    @pytest.mark.asyncio
    async def test_member_selection_prewarms_as_before(self):
        b = SimpleNamespace(
            execution_context=SimpleNamespace(selection_kind="member", member_id="m1"),
            selection_kind="member",
            kiro_agent="anthropic",
        )
        assert not await cb.blocks_prewarm(None, _slot(), KEY, _cfg({"anthropic": "cloud"}), b)


# ── Claude's own store under a profile's directory ───────────────────────────


class TestWalkers:
    def test_nothing_is_walked_by_default(self):
        assert dirs.walked_config_dir() is None

    def test_every_config_dir_runs_default_then_each_profile(self, signed_in):
        seen = []

        def probe(tag):
            seen.append((tag, dirs.walked_config_dir()))
            return True

        assert dirs.every_config_dir(probe)("x") is True
        assert seen == [("x", None), ("x", Path(signed_in))]
        assert dirs.walked_config_dir() is None

    def test_every_config_dir_is_false_when_any_run_fails(self, signed_in):
        assert dirs.every_config_dir(lambda: dirs.walked_config_dir() is None)() is False
        assert dirs.every_config_dir(lambda: dirs.walked_config_dir() is not None)() is False

    def test_every_config_dir_restores_after_an_error(self, signed_in):
        def boom():
            if dirs.walked_config_dir() is not None:
                raise OSError("unreadable")
            return True

        with pytest.raises(OSError):
            dirs.every_config_dir(boom)()
        assert dirs.walked_config_dir() is None

    def test_first_existing_prefers_the_default_then_a_profile(self, home, signed_in):
        def spell():
            return (dirs.walked_config_dir() or home / ".claude") / "projects" / "p" / SID

        wrapped = dirs.first_existing_config_dir(spell)
        default, cloud = (
            home / ".claude" / "projects" / "p" / SID,
            signed_in / "projects" / "p" / SID,
        )
        assert wrapped() == default  # nothing anywhere: the unwrapped answer
        cloud.mkdir(parents=True)
        assert wrapped() == cloud
        default.mkdir(parents=True)
        assert wrapped() == default
        assert dirs.first_existing_config_dir(lambda: None)() is None

    def test_session_config_dir_reads_the_clients_own_env(self):
        client = SimpleNamespace(_extra_env={"CLAUDE_CONFIG_DIR": "/h/.claude-cloud"})
        assert cp.session_config_dir(client) == Path("/h/.claude-cloud")
        assert cp.session_config_dir(SimpleNamespace(_extra_env={"AWS_PROFILE": "p"})) is None
        assert cp.session_config_dir(SimpleNamespace(_extra_env=None)) is None
        assert cp.session_config_dir(SimpleNamespace()) is None


# ── real runs: the production turn and prewarm, with only the provider scripted ─


class _ClaudeProvider(ScriptedProvider):
    capabilities = capabilities_for(_backends.ACP_BACKEND_CLAUDE)


def _turn_config(profiles, *, routes=None, default="claude"):
    local_ext = {"claude_profiles": {"agents": profiles}}
    if routes is not None:
        local_ext["chat_backends"] = {"agents": routes}
    return {"agent": {"acp_backend": default}, "local_ext": local_ext}


def _template_bindings(agent):
    return ResolvedBindings(
        workspace_dir=Path("workspace"),
        effective_memory_config={},
        kiro_agent=agent,
        model="",
        resolved_alias=agent,
        requested_resolved=True,
        memory_store_name="default",
        selection_kind="template",
        execution_context=ExecutionContext(
            None, MemoryStoreRef("default"), "template", agent, selection_name=agent
        ),
    )


def _error_rows(record):
    return [str(row.get("content")) for row in record.window if row.get("role") == "error"]


class TestRealRuns:
    @pytest.mark.asyncio
    async def test_a_profiled_turn_allocates_under_the_profile(self, signed_in):
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({TEMPLATE: "cloud"}),
        )
        assert [call.kwargs.get("agent") for call in record.allocations] == [TEMPLATE]
        assert record.allocations[0].kwargs.get("extra_env") == {
            "CLAUDE_CONFIG_DIR": str(signed_in)
        }
        assert "backend_override" not in record.allocations[0].kwargs
        assert _error_rows(record) == []

    @pytest.mark.asyncio
    async def test_an_unprofiled_turn_allocates_as_upstream_does(self, signed_in):
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({"another-template": "cloud"}),
        )
        assert [call.kwargs.get("agent") for call in record.allocations] == [TEMPLATE]
        assert "extra_env" not in record.allocations[0].kwargs
        assert "backend_override" not in record.allocations[0].kwargs

    @pytest.mark.asyncio
    async def test_a_profile_routed_to_claude_carries_both(self, signed_in):
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({TEMPLATE: "cloud"}, routes={TEMPLATE: "claude"}, default="codex"),
        )
        kwargs = record.allocations[0].kwargs
        assert kwargs.get("backend_override") == "claude"
        assert kwargs.get("extra_env") == {"CLAUDE_CONFIG_DIR": str(signed_in)}

    @pytest.mark.asyncio
    async def test_an_unsigned_profile_ends_the_turn_before_any_session(self, home):
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({TEMPLATE: "cloud"}),
        )
        assert record.allocations == []
        (row,) = _error_rows(record)
        assert row.startswith("claude_profile_unavailable:")
        assert 'CLAUDE_CONFIG_DIR="$HOME/.claude-cloud" claude' in row

    @pytest.mark.asyncio
    async def test_a_profile_on_another_harness_ends_the_turn_before_any_session(self, signed_in):
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({TEMPLATE: "cloud"}, routes={TEMPLATE: "codex"}),
        )
        assert record.allocations == []
        (row,) = _error_rows(record)
        assert row.startswith("claude_profile_unavailable:")
        assert "would not run on Claude Code" in row

    @pytest.mark.asyncio
    async def test_an_inherited_provider_variable_ends_the_turn_before_any_session(
        self, signed_in, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        record = await run_turn(
            TurnScript(message="hello", provider=_ClaudeProvider),
            config=_turn_config({TEMPLATE: "cloud"}),
        )
        assert record.allocations == []
        (row,) = _error_rows(record)
        assert "ANTHROPIC_BASE_URL" in row and "127.0.0.1" not in row

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profiles", "prewarmed"),
        [(None, 1), ({"another-template": "cloud"}, 1), ({"anthropic": "cloud"}, 0)],
    )
    async def test_the_prewarm_leaves_a_profiled_chat_to_its_first_turn(
        self, tmp_path, profiles, prewarmed
    ):
        slot = _ChatSlot("t1")
        slot.agent = "anthropic"
        slot.project = str(tmp_path)
        state = _mock_state(slot)
        state.sessions.get_mirror_link = MagicMock(return_value=None)
        loader = _eager_cfg(True)
        loader.return_value.agent.acp_backend = "claude"
        if profiles is not None:
            loader.return_value._extra_sections = {
                "local_ext": {"claude_profiles": {"agents": profiles}}
            }
        with (
            patch.object(chat_runner.KiroCrewConfig, "load", loader),
            patch.object(
                chat_runner, "resolve_agent_bindings", return_value=_template_bindings("anthropic")
            ),
        ):
            await chat_runner._eager_spawn(state, slot)
        assert state.sessions.get_or_create.await_count == prewarmed


# ── seam canaries: the upstream names this patch depends on ──────────────────


class TestUpstreamSeams:
    def test_allocation_takes_extra_env_and_bypasses_the_pool_for_it(self):
        from kiro_crew.session_allocation_parts import get_or_create as goc

        source = inspect.getsource(goc)
        assert "extra_env: dict[str, str] | None = None" in source
        assert 'pool_decision = "bypass_env"' in source
        assert "session.extra_env = dict(caller_env) if caller_env else None" in source

    def test_session_record_and_successor_identity_carry_extra_env(self):
        from kiro_crew.session_lifecycle_parts.parked_queue import allocation_identity
        from kiro_crew.session_parts.record import _Session

        assert "extra_env" in _Session.__dataclass_fields__
        assert '"extra_env": env' in inspect.getsource(allocation_identity)

    def test_launch_applies_extra_env_over_agent_env(self):
        from kiro_crew.acp import launch

        assert "extra_env" in launch.LaunchRequest.__dataclass_fields__
        source = inspect.getsource(launch)
        assert "env.update(request.extra_env)" in source
        assert source.index("tools.agent_env_overlay") < source.index(
            "env.update(request.extra_env)"
        )

    def test_the_spawn_scrub_keeps_the_config_directory(self):
        from kiro_crew.sandbox_spawn import scrub_agent_subprocess_env

        env = scrub_agent_subprocess_env({"CLAUDE_CONFIG_DIR": "/h/.claude-cloud", "PATH": "/bin"})
        assert env.get("CLAUDE_CONFIG_DIR") == "/h/.claude-cloud"

    def test_claude_runs_one_process_per_session(self):
        # A shared process could not hold two config directories.
        from kiro_crew.acp.harness import _PROCESS_ADAPTERS
        from kiro_crew.agent_sdk.backends import ACP_BACKENDS_SESSION_SHARING

        assert _backends.ACP_BACKEND_CLAUDE in _PROCESS_ADAPTERS
        assert _backends.ACP_BACKEND_CLAUDE not in ACP_BACKENDS_SESSION_SHARING
