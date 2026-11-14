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
