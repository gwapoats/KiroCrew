"""LOCAL PATCH: ``kiro_crew.local_ext.chat_backends`` -- dashboard chats routed by template.

Unit tests of the module's decisions, plus "seam canaries" that pin the upstream
names the patch leans on, so a rebase onto a release that moved one fails here
first rather than silently routing nothing.
"""

from __future__ import annotations

import inspect
import json
import logging
from types import SimpleNamespace

import pytest

from kiro_crew.agent_sdk import backends as _backends
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable
from kiro_crew.local_ext import chat_backends as cb

TEMPLATE_KEY = "dashboard:chat-1"


def _cfg(agents=None, *, configured=True, default="claude", member="kas", agents_cfg=None):
    extra = {}
    if configured:
        feature = {} if agents is None else {"agents": agents}
        extra = {"local_ext": {"chat_backends": feature}}
    return SimpleNamespace(
        _extra_sections=extra,
        agent=SimpleNamespace(acp_backend=default, member_acp_backend=member, kas_bundle=False),
        agents=agents_cfg or {},
    )


def _slot(**over):
    base = dict(
        key="chat-1",
        agent="codex-coder",
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


def _template_exec(name="codex-coder"):
    return SimpleNamespace(selection_kind="template", member_id=None, template_id=name)


class _Sessions:
    def __init__(self, provider=None, override=None, mirror=None):
        self._provider = provider
        self._mirror = mirror
        self._sessions = {TEMPLATE_KEY: SimpleNamespace(backend_override=override)}

    def _fold_key(self, key):
        return key

    def get_provider(self, key):
        return self._provider

    def get_mirror_link(self, key):
        return self._mirror


def _provider(backend):
    return SimpleNamespace(capabilities=capabilities_for(backend))


@pytest.fixture(autouse=True)
def _selectable(monkeypatch):
    """A deterministic selectable set, whatever this host registered."""
    monkeypatch.setattr(
        _backends,
        "_selectable",
        {"", "kas", "claude", "codex"},
    )
    monkeypatch.setattr(_backends, "_host_unservable", {})


@pytest.fixture
def armed(monkeypatch):
    calls = []
    from kiro_crew.dashboard import session_control

    monkeypatch.setattr(session_control, "arm_model_switch_reset", lambda st, sl: calls.append(sl))
    return calls


# ── configuration ────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_absent_section_is_off(self):
        assert cb.feature_section(_cfg(configured=False)) is None
        assert cb.agent_routes(_cfg(configured=False)) is None

    @pytest.mark.parametrize("bad", ["codex", ["codex"], 3, None])
    def test_non_object_section_is_off(self, bad):
        cfg = _cfg(configured=False)
        cfg._extra_sections = {"local_ext": {"chat_backends": bad}}
        assert cb.agent_routes(cfg) is None

    def test_configured_without_map_is_empty(self):
        assert cb.agent_routes(_cfg(None)) == {}

    def test_route_lookup(self):
        cfg = _cfg({"codex-coder": "codex", "plain": ""})
        assert cb.route_for_template(cfg, "codex-coder") == "codex"
        assert cb.route_for_template(cfg, "plain") is None
        assert cb.route_for_template(cfg, "other") is None
        assert cb.route_for_template(cfg, None) is None

    def test_kiro_reads_like_a_stored_acp_backend(self, monkeypatch):
        monkeypatch.setattr(_backends, "_kiro_cli_offers_kas", True)
        assert cb.route_for_template(_cfg({"k": "kiro"}), "k") == "kas"
        monkeypatch.setattr(_backends, "_kiro_cli_offers_kas", False)
        assert cb.route_for_template(_cfg({"k": "kiro"}), "k") == ""

    @pytest.mark.parametrize("stored", ["goose", "sk-secret-token", 7, ["codex"]])
    def test_unstartable_route_fails_closed_without_echoing(self, stored):
        with pytest.raises(cb.ChatBackendUnavailable) as info:
            cb.route_for_template(_cfg({"codex-coder": stored}), "codex-coder")
        exc = info.value
        assert isinstance(exc, CrewmateHarnessUnavailable)
        assert exc.code == "chat_backend_unavailable"
        assert "codex-coder" in str(exc)
        assert "sk-secret-token" not in str(exc)

    def test_denied_by_governance_fails_closed(self, monkeypatch):
        monkeypatch.setattr(_backends, "_selectable", {"", "claude"})
        with pytest.raises(cb.ChatBackendUnavailable):
            cb.route_for_template(_cfg({"codex-coder": "codex"}), "codex-coder")


# ── which chats are routable ─────────────────────────────────────────────────


class TestRoutable:
    def test_ordinary_chat(self):
        assert cb.slot_is_routable(SimpleNamespace(sessions=_Sessions()), _slot(), TEMPLATE_KEY)

    @pytest.mark.parametrize(
        "over",
        [
            {"mode": "dm"},
            {"_app": "someapp"},
            {"app_agent_owner": "app/agent"},
            {"linked_session_key": "slack:123.4"},
            {"_slack_channel": "C1"},
            {"channel_origin": True},
        ],
    )
    def test_excluded_slots(self, over, monkeypatch):
        from kiro_crew import members

        monkeypatch.setattr(members, "DM_SLOT_MODE", "dm", raising=False)
        state = SimpleNamespace(sessions=_Sessions())
        assert not cb.slot_is_routable(state, _slot(**over), TEMPLATE_KEY)

    def test_bot_dm_key_is_excluded(self):
        assert not cb.slot_is_routable(None, _slot(), "dashboard:member-atlas")

    def test_bot_worker_is_excluded(self, monkeypatch):
        from kiro_crew import members

        members.note_crewmate_worker_session(TEMPLATE_KEY, "m1")
        try:
            assert not cb.slot_is_routable(None, _slot(), TEMPLATE_KEY)
        finally:
            members.forget_crewmate_worker_session(TEMPLATE_KEY)

    def test_mirrored_chat_is_excluded(self):
        state = SimpleNamespace(sessions=_Sessions(mirror=object()))
        assert not cb.slot_is_routable(state, _slot(), TEMPLATE_KEY)

    def test_execution_kinds(self):
        assert cb.execution_is_template(_template_exec())
        assert not cb.execution_is_template(
            SimpleNamespace(selection_kind="member", member_id="m1")
        )
        # A Bot's worker: a template selection that keeps its member id.
        assert not cb.execution_is_template(
            SimpleNamespace(selection_kind="template", member_id="m1")
        )
        assert not cb.execution_is_template(None)
        assert cb.execution_is_template(None, "template")


# ── turn admission ───────────────────────────────────────────────────────────


class TestTurnRoute:
    @pytest.mark.asyncio
    async def test_unconfigured_changes_nothing(self, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("codex"), override="codex"))
        got = await cb.turn_route(
            state, _slot(), TEMPLATE_KEY, _cfg(configured=False), "codex-coder", _template_exec()
        )
        assert got == {} and armed == []

    @pytest.mark.asyncio
    async def test_no_config_object(self, armed):
        got = await cb.turn_route(None, _slot(), TEMPLATE_KEY, None, "x", _template_exec())
        assert got == {}

    @pytest.mark.asyncio
    async def test_routed_template_gets_backend_override(self, armed):
        state = SimpleNamespace(sessions=_Sessions())
        got = await cb.turn_route(
            state,
            _slot(),
            TEMPLATE_KEY,
            _cfg({"codex-coder": "codex"}),
            "codex-coder",
            _template_exec(),
        )
        assert got == {"backend_override": "codex"}
        assert armed == []

    @pytest.mark.asyncio
    async def test_route_equal_to_default_keeps_the_pool(self, armed):
        state = SimpleNamespace(sessions=_Sessions())
        got = await cb.turn_route(
            state,
            _slot(),
            TEMPLATE_KEY,
            _cfg({"codex-coder": "claude"}),
            "codex-coder",
            _template_exec(),
        )
        assert got == {}

    @pytest.mark.asyncio
    async def test_member_selection_is_not_routed(self, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        member = SimpleNamespace(selection_kind="member", member_id="m1")
        got = await cb.turn_route(
            state, _slot(), TEMPLATE_KEY, _cfg({"codex-coder": "codex"}), "codex-coder", member
        )
        assert got == {} and armed == []

    @pytest.mark.asyncio
    async def test_broken_route_raises_before_any_claim(self, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        with pytest.raises(cb.ChatBackendUnavailable):
            await cb.turn_route(
                state,
                _slot(),
                TEMPLATE_KEY,
                _cfg({"codex-coder": "goose"}),
                "codex-coder",
                _template_exec(),
            )
        assert armed == []

    @pytest.mark.asyncio
    async def test_live_session_on_default_is_reset_onto_the_route(self, armed):
        slot = _slot()
        state = SimpleNamespace(sessions=_Sessions(_provider("claude")))
        got = await cb.turn_route(
            state,
            slot,
            TEMPLATE_KEY,
            _cfg({"codex-coder": "codex"}),
            "codex-coder",
            _template_exec(),
        )
        assert got == {"backend_override": "codex"}
        assert armed == [slot]

    @pytest.mark.asyncio
    async def test_live_session_already_on_the_route_is_kept(self, armed):
        state = SimpleNamespace(sessions=_Sessions(_provider("codex"), override="codex"))
        await cb.turn_route(
            state,
            _slot(),
            TEMPLATE_KEY,
            _cfg({"codex-coder": "codex"}),
            "codex-coder",
            _template_exec(),
        )
        assert armed == []

    @pytest.mark.asyncio
    async def test_route_removed_moves_our_session_back(self, armed):
        slot = _slot()
        state = SimpleNamespace(sessions=_Sessions(_provider("codex"), override="codex"))
        got = await cb.turn_route(
            state, slot, TEMPLATE_KEY, _cfg({}), "codex-coder", _template_exec()
        )
        assert got == {}
        assert armed == [slot]

    @pytest.mark.asyncio
    async def test_unrouted_unstamped_session_keeps_upstream_behaviour(self, armed):
        # e.g. agent.acp_backend changed in Settings: upstream leaves live sessions alone.
        state = SimpleNamespace(sessions=_Sessions(_provider("codex"), override=None))
        await cb.turn_route(state, _slot(), TEMPLATE_KEY, _cfg({}), "codex-coder", _template_exec())
        assert armed == []

    @pytest.mark.asyncio
    async def test_provider_without_capabilities_is_left_alone(self, armed):
        state = SimpleNamespace(sessions=_Sessions(SimpleNamespace()))
        await cb.turn_route(
            state,
            _slot(),
            TEMPLATE_KEY,
            _cfg({"codex-coder": "codex"}),
            "codex-coder",
            _template_exec(),
        )
        assert armed == []


class TestPrewarm:
    @pytest.mark.asyncio
    async def test_routed_chat_is_not_prewarmed(self):
        b = SimpleNamespace(
            execution_context=None, selection_kind="template", kiro_agent="codex-coder"
        )
        assert await cb.blocks_prewarm(
            None, _slot(), TEMPLATE_KEY, _cfg({"codex-coder": "codex"}), b
        )

    @pytest.mark.asyncio
    async def test_broken_route_is_not_prewarmed(self):
        b = SimpleNamespace(
            execution_context=None, selection_kind="template", kiro_agent="codex-coder"
        )
        assert await cb.blocks_prewarm(
            None, _slot(), TEMPLATE_KEY, _cfg({"codex-coder": "goose"}), b
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cfg",
        [
            _cfg(configured=False),
            _cfg({}),
            _cfg({"codex-coder": "claude"}),
            _cfg({"other": "codex"}),
        ],
    )
    async def test_everything_else_prewarms_as_before(self, cfg):
        b = SimpleNamespace(
            execution_context=None, selection_kind="template", kiro_agent="codex-coder"
        )
        assert not await cb.blocks_prewarm(None, _slot(), TEMPLATE_KEY, cfg, b)

    @pytest.mark.asyncio
    async def test_member_selection_prewarms_as_before(self):
        b = SimpleNamespace(
            execution_context=SimpleNamespace(selection_kind="member", member_id="m1"),
            selection_kind="member",
            kiro_agent="codex-coder",
        )
        assert not await cb.blocks_prewarm(
            None, _slot(), TEMPLATE_KEY, _cfg({"codex-coder": "codex"}), b
        )


class TestColdSlot:
    @pytest.mark.asyncio
    async def test_cold_slot_override(self, monkeypatch):
        from kiro_crew import session_agent_selection

        bindings = SimpleNamespace(
            execution_context=None, selection_kind="template", kiro_agent="codex-coder"
        )
        monkeypatch.setattr(
            session_agent_selection, "resolve_session_agent_bindings", lambda *a, **k: bindings
        )
        cfg = _cfg({"codex-coder": "codex"})
        assert await cb.slot_backend_override(cfg, _slot(), TEMPLATE_KEY) == "codex"
        assert await cb.slot_backend_override(_cfg({}), _slot(), TEMPLATE_KEY) is None
        assert await cb.slot_backend_override(_cfg(configured=False), _slot(), TEMPLATE_KEY) is None

    @pytest.mark.asyncio
    async def test_cold_slot_unresolvable_is_none(self, monkeypatch):
        from kiro_crew import session_agent_selection

        def boom(*a, **k):
            raise RuntimeError("unreadable")

        monkeypatch.setattr(session_agent_selection, "resolve_session_agent_bindings", boom)
        cfg = _cfg({"codex-coder": "codex"})
        assert await cb.slot_backend_override(cfg, _slot(), TEMPLATE_KEY) is None

    @pytest.mark.asyncio
    async def test_catalog_names_the_live_harness_of_a_routed_chat(self):
        slot = _slot()
        state = SimpleNamespace(sessions=_Sessions(_provider("codex")), _slots={"chat-1": slot})
        got = await cb.catalog_session_backend(state, _cfg({"codex-coder": "codex"}), TEMPLATE_KEY)
        assert got == "codex"
        assert await cb.catalog_session_backend(state, _cfg(configured=False), TEMPLATE_KEY) is None


# ── config round trip and the validation registration ───────────────────────


class TestConfigSection:
    def _load(self, tmp_path, monkeypatch, data):
        from kiro_crew.config import loader as L

        cfgp = tmp_path / "config.json"
        cfgp.write_text(json.dumps(data))
        monkeypatch.setattr(L, "config_path", lambda: cfgp)
        monkeypatch.setattr(L, "config_dir", lambda: tmp_path)
        monkeypatch.setattr(L, "config_local_path", lambda: tmp_path / "config.local.json")
        return L.KiroCrewConfig.load()

    def test_section_survives_load_and_save_and_is_read(self, tmp_path, monkeypatch, caplog):
        section = {"chat_backends": {"agents": {"codex-coder": "codex"}}}
        with caplog.at_level(logging.WARNING):
            cfg = self._load(
                tmp_path, monkeypatch, {"agent": {"provider": "acp"}, "local_ext": section}
            )
        assert not [r for r in caplog.records if "unrecognized top-level keys" in r.getMessage()]
        assert cfg.to_dict()["local_ext"] == section
        assert cb.agent_routes(cfg) == {"codex-coder": "codex"}

    def test_scalar_section_still_warns(self, caplog):
        from kiro_crew.config import validation

        if not validation._HAS_JSONSCHEMA:
            pytest.skip("jsonschema not installed")
        with caplog.at_level(logging.WARNING, logger="kiro_crew.config.loader"):
            validation.validate_config_data({"agent": {"provider": "acp"}, "local_ext": "x"})
        assert any("local_ext" in r.getMessage() for r in caplog.records)


# ── seam canaries: the upstream names this patch depends on ──────────────────


class TestUpstreamSeams:
    def test_gate_takes_backend_override_and_crew_backend(self):
        from kiro_crew.members import select_provider_backend

        params = inspect.signature(select_provider_backend).parameters
        assert "backend_override" in params and "crew_backend" in params

    def test_session_record_and_successor_identity_carry_the_override(self):
        from kiro_crew.session_lifecycle_parts.parked_queue import allocation_identity
        from kiro_crew.session_parts.record import _Session

        assert "backend_override" in _Session.__dataclass_fields__
        assert "backend_override" in inspect.getsource(allocation_identity)

    def test_session_model_and_inherited_default_take_the_override(self):
        from kiro_crew.dashboard.chat_turn.model_fallback import _inherited_default_model
        from kiro_crew.session import _session_model

        assert "backend_override" in inspect.signature(_session_model).parameters
        assert "backend_override" in inspect.signature(_inherited_default_model).parameters

    def test_session_manager_lookup_seam(self):
        from kiro_crew.session import SessionManager

        assert callable(getattr(SessionManager, "_fold_key", None))

    def test_call_sites_are_present(self):
        from pathlib import Path

        root = Path(cb.__file__).resolve().parents[1]
        for rel, needle in [
            ("dashboard/chat_turn/turn_admission.py", "turn_route("),
            ("dashboard/chat_turn/eager_spawn.py", "blocks_prewarm("),
            ("dashboard/chat_turn/model_fallback.py", "backend_override=backend_override"),
            ("config/validation.py", '"local_ext"'),
        ]:
            assert needle in (root / rel).read_text(encoding="utf-8"), rel
