"""LOCAL PATCH (not upstream): each Claude profile's model list stays its own.

A ``~/.claude`` session (local model) and a ``~/.claude-cloud`` session
(subscription) both advertise ``claude_code`` ids. The picker must list the
profile the chat's next turn runs on: the cloud chat must not show the local
list before its first turn, and a local chat must not show the cloud list after.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kiro_crew import model_registry
from kiro_crew.dashboard.handlers.agents import _advertised_cc_models
from kiro_crew.local_ext import claude_profile_models as m
from kiro_crew.local_ext.claude_profile_dirs import CONFIG_DIR_ENV, config_dir

LOCAL = [{"modelId": "Qwen3.8-Flash-Next-oQ6e-mtp-1m", "name": "Qwen3.8"}]
CLOUD = [{"modelId": "opus", "name": "Opus 5.5"}]


def _provider(models, profile=None):
    from kiro_crew.acp_backends import ACP_BACKEND_CLAUDE
    from kiro_crew.agent_sdk.capabilities import capabilities_for
    from kiro_crew.providers.acp import AcpProvider

    provider = MagicMock(spec=AcpProvider)
    provider.capabilities = capabilities_for(ACP_BACKEND_CLAUDE)
    provider.available_models.return_value = models
    env = {CONFIG_DIR_ENV: config_dir(profile)} if profile else {}
    provider.client = SimpleNamespace(_extra_env=env)
    return provider


def _request(*providers):
    sessions = SimpleNamespace(active_providers=lambda: list(providers))
    req = MagicMock()
    req.app.__getitem__.return_value = SimpleNamespace(sessions=sessions)
    return req


@pytest.fixture
def clean_cache(monkeypatch):
    monkeypatch.setattr(model_registry, "_ADVERTISED_MODELS", {})
    monkeypatch.setattr(model_registry, "_ADVERTISED_META", {})
    monkeypatch.setattr(model_registry, "_PENDING_ADVERTISED_SHRINK", {})


def _ids(rows):
    return [r["model_name"] for r in rows]


class TestProfileOfSession:
    def test_profile_read_off_config_dir(self):
        assert m.provider_profile(_provider(CLOUD, "cloud")) == "cloud"
        assert m.provider_profile(_provider(LOCAL)) is None

    def test_unknown_dir_is_no_profile(self):
        p = _provider(LOCAL)
        p.client = SimpleNamespace(_extra_env={CONFIG_DIR_ENV: "/elsewhere"})
        assert m.provider_profile(p) is None

    def test_session_handle_reads_runtime_env(self):
        handle = SimpleNamespace(
            _runtime=SimpleNamespace(_extra_env={CONFIG_DIR_ENV: config_dir("cloud")})
        )
        assert m.owner_profile(handle) == "cloud"

    def test_tilde_spelling_matches(self):
        home = os.path.expanduser("~")
        assert m.profile_of_dir(config_dir("cloud").replace(home, "~", 1)) == "cloud"


class TestModelsBackendSpelling:
    def test_split(self):
        assert m.split_models_backend("claude@cloud") == ("claude", "cloud")
        assert m.split_models_backend("claude") == ("claude", None)
        assert m.split_models_backend("claude@nope") == ("claude@nope", None)
        assert m.split_models_backend(None) == (None, None)

    def test_query_rewritten(self):
        query, profile = m.query_without_profile({"backend": "claude@cloud", "x": "1"})
        assert (query, profile) == ({"backend": "claude", "x": "1"}, "cloud")

    def test_query_untouched_without_profile(self):
        q = {"backend": "codex"}
        assert m.query_without_profile(q) == (q, None)

    def test_models_backend_for(self):
        assert m.models_backend_for("claude", "cloud") == "claude@cloud"
        assert m.models_backend_for("claude", None) is None


class TestPickerFilter:
    def test_cloud_chat_skips_newer_local_session(self, clean_cache):
        req = _request(_provider(CLOUD, "cloud"), _provider(LOCAL))
        assert _ids(_advertised_cc_models(req, "claude_code", profile="cloud")) == ["opus"]

    def test_local_chat_skips_newer_cloud_session(self, clean_cache):
        req = _request(_provider(LOCAL), _provider(CLOUD, "cloud"))
        assert _ids(_advertised_cc_models(req, "claude_code")) == [LOCAL[0]["modelId"]]

    def test_cloud_chat_cold_reads_its_own_snapshot(self, clean_cache):
        model_registry.refresh_advertised_models("claude_code", ["Qwen-local"])
        model_registry.refresh_advertised_models(
            m.snapshot_key("claude_code", "cloud"), ["opus"], rows=CLOUD
        )
        rows = _advertised_cc_models(_request(_provider(LOCAL)), "claude_code", profile="cloud")
        assert rows == [{"model_name": "opus", "display_name": "Opus 5.5", "description": ""}]

    def test_no_profile_snapshot_falls_back_to_upstream_key(self, clean_cache):
        model_registry.refresh_advertised_models("claude_code", ["shared"])
        assert _ids(_advertised_cc_models(_request(), "claude_code", profile="cloud")) == ["shared"]

    def test_without_profiles_every_session_answers(self, clean_cache):
        # Upstream behaviour: the newest session wins, whatever it advertised.
        req = _request(_provider(CLOUD), _provider(LOCAL))
        assert _ids(_advertised_cc_models(req, "claude_code")) == [LOCAL[0]["modelId"]]


class TestRecordSnapshot:
    def test_profiled_session_recorded_under_its_key(self, clean_cache):
        owner = SimpleNamespace(_extra_env={CONFIG_DIR_ENV: config_dir("cloud")})
        assert m.record_snapshot(owner, "claude_code", ["opus"], CLOUD) is True
        assert model_registry.advertised_models("claude_code@cloud") == ["opus"]

    def test_default_session_recorded_once_profiles_exist(self, clean_cache, monkeypatch):
        monkeypatch.setattr(m, "existing_config_dirs", lambda: (config_dir("cloud"),))
        assert m.record_snapshot(SimpleNamespace(_extra_env={}), "claude_code", ["q"], None)
        assert model_registry.advertised_models("claude_code@default") == ["q"]

    def test_inert_without_profiles(self, clean_cache, monkeypatch):
        monkeypatch.setattr(m, "existing_config_dirs", lambda: ())
        assert not m.record_snapshot(SimpleNamespace(_extra_env={}), "claude_code", ["q"], None)
        assert model_registry.advertised_models("claude_code@default") == []

    def test_other_namespaces_ignored(self, clean_cache):
        owner = SimpleNamespace(_extra_env={CONFIG_DIR_ENV: config_dir("cloud")})
        assert not m.record_snapshot(owner, "codex", ["gpt"], None)


class TestSlotProfile:
    @pytest.mark.asyncio
    async def test_off_without_config(self):
        cfg = SimpleNamespace(_extra_sections={})
        assert await m.slot_profile(None, SimpleNamespace(), "k", cfg) is None

    @pytest.mark.asyncio
    async def test_template_profile_even_while_local_session_live(self, monkeypatch):
        from kiro_crew.local_ext import chat_backends

        cfg = SimpleNamespace(
            _extra_sections={"local_ext": {"claude_profiles": {"agents": {"claude": "cloud"}}}}
        )

        async def routable(*_a):
            return True

        monkeypatch.setattr(chat_backends, "_routable", routable)
        monkeypatch.setattr(m, "_cold_slot_profile", lambda c, s, k: "cloud")
        state = SimpleNamespace(sessions=SimpleNamespace(get_provider=lambda k: _provider(LOCAL)))
        assert await m.slot_profile(state, SimpleNamespace(agent="claude"), "k", cfg) == "cloud"


class TestProfileModelsField:
    @pytest.mark.asyncio
    async def test_profiled_chat_gets_profile_list(self, monkeypatch):
        async def cloud(*_a):
            return "cloud"

        monkeypatch.setattr(m, "slot_profile", cloud)
        slot = SimpleNamespace(key="k", linked_session_key="")
        out = await m.profile_models_field(None, slot, {}, config=object())
        assert out == {"models_backend": "claude@cloud"}

    @pytest.mark.asyncio
    async def test_unprofiled_chat_keeps_upstream_field(self, monkeypatch):
        async def none(*_a):
            return None

        monkeypatch.setattr(m, "slot_profile", none)
        field = {"models_backend": "codex"}
        out = await m.profile_models_field(None, SimpleNamespace(key="k"), field, config=object())
        assert out == field

    @pytest.mark.asyncio
    async def test_failure_keeps_upstream_field(self, monkeypatch):
        async def boom(*_a):
            raise RuntimeError("x")

        monkeypatch.setattr(m, "slot_profile", boom)
        out = await m.profile_models_field(None, SimpleNamespace(key="k"), {}, config=object())
        assert out == {}


# ── a model stays with the profile that serves it ────────────────────────────

QWEN = "Qwen3.8-Flash-Next-oQ6e-mtp-1m"
# What the two config directories advertised on the machine this was written for:
# each list carries the other profile's own model as a row named by its id.
LOCAL_ROWS = [
    {"modelId": "default", "name": "Default (recommended)"},
    {"modelId": QWEN, "name": f"{QWEN} ({QWEN})"},
    {"modelId": "opus", "name": f"{QWEN} (opus)"},
    {"modelId": "claude-fable-5-2", "name": "claude-fable-5-2"},
    {"modelId": "claude-opus-4-8", "name": "Opus 4.8"},
]
CLOUD_ROWS = [
    {"modelId": "default", "name": "Default (recommended)"},
    {"modelId": QWEN, "name": QWEN},
    {"modelId": "opus", "name": "Opus 5.5"},
    {"modelId": "claude-fable-5-2", "name": "Fable 5.2"},
    {"modelId": "claude-opus-4-8", "name": "Opus 4.8"},
]


def _row_ids(rows):
    return [r["modelId"] for r in rows]


@pytest.fixture
def both_profiles(clean_cache):
    for profile, rows in ((None, LOCAL_ROWS), ("cloud", CLOUD_ROWS)):
        model_registry.refresh_advertised_models(
            m.snapshot_key("claude_code", profile), _row_ids(rows), rows=rows
        )


def _client(model, profile=None, backend="claude", tmp_path=None):
    from kiro_crew.acp.client import AcpClient

    env = {CONFIG_DIR_ENV: config_dir(profile)} if profile else None
    return AcpClient(work_dir=str(tmp_path), model=model, extra_env=env, acp_backend=backend)


class TestNativeIds:
    def test_a_row_named_by_its_id_is_not_the_profiles_own(self, both_profiles):
        assert m.native_ids(None) == {"default", QWEN, "opus", "claude-opus-4-8"}
        assert m.native_ids("cloud") == {"default", "opus", "claude-fable-5-2", "claude-opus-4-8"}

    def test_empty_before_the_profile_advertised(self, clean_cache):
        assert m.native_ids("cloud") == frozenset()

    def test_the_fold_it_matches_ids_with_is_the_registrys(self):
        fold = model_registry._normalize_advertised_key
        assert fold("claude-opus-4.8") == fold("claude-opus-4-8")
        assert fold("global.anthropic.claude-opus-5[1m]") == fold("claude-opus-5")
        assert fold(QWEN) != fold("opus")


class TestForeignTo:
    def test_the_other_profiles_own_model(self, both_profiles):
        assert m.foreign_to(QWEN, "cloud") == "default"
        assert m.foreign_to("claude-fable-5-2", None) == "cloud"

    def test_a_model_both_serve_is_nobodys_to_refuse(self, both_profiles):
        assert m.foreign_to("opus", "cloud") is None
        assert m.foreign_to("opus", None) is None
        assert m.foreign_to("claude-opus-4.8", None) is None  # another spelling of its own id

    def test_a_model_no_profile_claims(self, both_profiles):
        assert m.foreign_to("claude-opus-5-5", "cloud") is None

    def test_a_profile_with_no_list_yet_does_not_take_the_others_model(self, clean_cache):
        model_registry.refresh_advertised_models(
            m.snapshot_key("claude_code", None), _row_ids(LOCAL_ROWS), rows=LOCAL_ROWS
        )
        assert m.foreign_to(QWEN, "cloud") == "default"
        assert m.foreign_to("claude-opus-5-5", "cloud") is None  # nobody's: sent as asked
        assert m.foreign_to(QWEN, None) is None

    def test_inert_while_no_other_profile_has_a_list(self, clean_cache):
        model_registry.refresh_advertised_models(
            m.snapshot_key("claude_code", None), _row_ids(LOCAL_ROWS), rows=LOCAL_ROWS
        )
        assert m.foreign_to("claude-fable-5-2", None) is None
        assert m.foreign_to("anything", None) is None


class TestClientModel:
    def test_auto_is_the_config_default(self):
        from kiro_crew.config.sections import DEFAULT_MODEL

        assert m.AUTO == DEFAULT_MODEL

    def test_a_cloud_session_is_not_started_on_the_local_model(self, both_profiles, tmp_path):
        assert _client(QWEN, "cloud", tmp_path=tmp_path)._model == m.AUTO

    def test_a_local_session_is_not_started_on_a_cloud_only_model(self, both_profiles, tmp_path):
        assert _client("claude-fable-5-2", tmp_path=tmp_path)._model == m.AUTO

    @pytest.mark.parametrize(
        "model, profile",
        [(QWEN, None), ("opus", "cloud"), ("opus", None), ("claude-fable-5-2", "cloud")],
    )
    def test_its_own_model_is_kept(self, both_profiles, tmp_path, model, profile):
        assert _client(model, profile, tmp_path=tmp_path)._model == model

    def test_another_harness_is_left_alone(self, both_profiles, tmp_path):
        assert _client("claude-fable-5-2", backend="codex", tmp_path=tmp_path)._model == (
            "claude-fable-5-2"
        )

    def test_inert_without_a_profile_list(self, clean_cache, tmp_path):
        assert _client(QWEN, "cloud", tmp_path=tmp_path)._model == QWEN
        assert _client("claude-fable-5-2", tmp_path=tmp_path)._model == "claude-fable-5-2"

    def test_the_first_session_of_a_profile_is_not_started_on_the_local_model(
        self, clean_cache, tmp_path
    ):
        model_registry.refresh_advertised_models(
            m.snapshot_key("claude_code", None), _row_ids(LOCAL_ROWS), rows=LOCAL_ROWS
        )
        assert _client(QWEN, "cloud", tmp_path=tmp_path)._model == m.AUTO

    def test_a_failure_keeps_the_model(self, both_profiles, tmp_path, monkeypatch):
        def boom(*_a):
            raise RuntimeError("x")

        monkeypatch.setattr(m, "foreign_to", boom)
        assert _client(QWEN, "cloud", tmp_path=tmp_path)._model == QWEN


class TestPinAppliesToARunningSession:
    """``model_scope.pin_applies`` with the session's own list (a claimed warm process)."""

    LOCAL_LIST = ["default", QWEN, "opus", "claude-opus-4-8"]

    def test_a_cloud_only_model_is_not_applied_to_a_local_session(self, both_profiles):
        from kiro_crew import model_scope

        assert not model_scope.pin_applies(
            "claude-fable-5-2", "claude_code", advertised=self.LOCAL_LIST
        )

    @pytest.mark.parametrize("model", ["opus", QWEN, "claude-opus-4.8", "claude-opus-5-5"])
    def test_other_models_get_upstreams_answer(self, both_profiles, model):
        assert not m.foreign_to_session(model, "claude_code", self.LOCAL_LIST)

    def test_untouched_without_a_named_profiles_list(self, clean_cache):
        assert not m.foreign_to_session("claude-fable-5-2", "claude_code", self.LOCAL_LIST)

    def test_untouched_without_the_sessions_list_or_on_another_harness(self, both_profiles):
        assert not m.foreign_to_session("claude-fable-5-2", "claude_code", None)
        assert not m.foreign_to_session("claude-fable-5-2", "codex", self.LOCAL_LIST)


class TestPickerLeavesOutTheOtherProfilesModel:
    def test_a_live_cloud_session_that_echoes_the_local_model(self, both_profiles):
        req = _request(_provider(CLOUD_ROWS, "cloud"))
        rows = _advertised_cc_models(req, "claude_code", profile="cloud")
        assert _ids(rows) == ["default", "opus", "claude-fable-5-2", "claude-opus-4-8"]

    def test_a_live_local_session_that_echoes_a_cloud_model(self, both_profiles):
        rows = _advertised_cc_models(_request(_provider(LOCAL_ROWS)), "claude_code")
        assert _ids(rows) == ["default", QWEN, "opus", "claude-opus-4-8"]

    def test_the_cold_snapshot_too(self, both_profiles):
        rows = _advertised_cc_models(_request(), "claude_code", profile="cloud")
        assert QWEN not in _ids(rows) and "claude-fable-5-2" in _ids(rows)

    def test_a_custom_model_no_profile_claims_stays(self, both_profiles):
        custom = [*CLOUD_ROWS, {"modelId": "claude-opus-5-5", "name": "claude-opus-5-5"}]
        rows = _advertised_cc_models(
            _request(_provider(custom, "cloud")), "claude_code", profile="cloud"
        )
        assert "claude-opus-5-5" in _ids(rows) and QWEN not in _ids(rows)

    def test_other_namespaces_are_not_read(self, both_profiles):
        rows = [{"model_name": QWEN, "display_name": QWEN}]
        assert m.own_rows(rows, "cloud", "codex") == rows
