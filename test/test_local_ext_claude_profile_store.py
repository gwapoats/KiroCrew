"""LOCAL PATCH: a Claude profile's config directory is fenced and cleaned as ``~/.claude`` is.

The upstream functions that fence Claude's token and settings files, delete a
disposable session's transcripts and read a session's user settings all resolve
``~/.claude``. These tests cover the hooks that make each of them reach a
profile's directory too, and pin the upstream names those hooks lean on.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiro_crew.local_ext import claude_profile_dirs as dirs
from kiro_crew.local_ext import claude_profiles as cp

SID = "0b6f1c1e-3a52-4c8d-9a55-6f0d2c1e7a10"


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


# ── the fences ───────────────────────────────────────────────────────────────


class TestFences:
    def test_listed_exactly_as_the_default_config_is(self):
        from kiro_crew.security import paths_lists

        for leaf in dirs.CREDENTIAL_LEAVES:
            assert leaf in paths_lists._SENSITIVE_HOME_DIRS
            assert leaf not in paths_lists._WRITE_PROTECTED_HOME_PATHS
        for leaf in dirs.SETTINGS_LEAVES:
            assert leaf in paths_lists._WRITE_PROTECTED_HOME_PATHS
            assert leaf not in paths_lists._SENSITIVE_HOME_DIRS
        # The twins they mirror are still where upstream put them.
        assert ".claude/.credentials.json" in paths_lists._SENSITIVE_HOME_DIRS
        assert ".claude/settings.json" in paths_lists._WRITE_PROTECTED_HOME_PATHS

    def test_the_gates_refuse_what_they_refuse_under_the_default_config(self):
        from kiro_crew.security import is_sensitive_path, is_sensitive_write_path

        real_home = os.path.expanduser("~")

        def verdict(rel):
            path = os.path.join(real_home, *rel.split("/"))
            return is_sensitive_path(path), is_sensitive_write_path(path)

        for name in (".credentials.json", "settings.json", "settings.local.json"):
            assert verdict(f".claude-cloud/{name}") == verdict(f".claude/{name}"), name
        assert verdict(".claude-cloud/.credentials.json") == (True, True)
        assert verdict(".claude-cloud/settings.json") == (False, True)
        # Transcripts and auto-memory stay as reachable as they are under ``~/.claude``.
        assert verdict(".claude-cloud/projects/p/memory/MEMORY.md") == (False, False)

    def test_the_token_file_is_in_the_mask_an_enforced_harness_gets(self):
        from kiro_crew.security import sandbox_credential_targets

        tail = os.path.join(".claude-cloud", ".credentials.json")
        assert any(target.endswith(tail) for target in sandbox_credential_targets())

    def test_the_declaration_tables_carry_the_leaves_once(self):
        from kiro_crew.agent_sdk import host_auth, native_allow_stores

        credentials, settings = host_auth.credential_leaves(), native_allow_stores.home_leaves()
        for leaf in dirs.CREDENTIAL_LEAVES:
            assert credentials.count(leaf) == 1
        for leaf in dirs.SETTINGS_LEAVES:
            assert settings.count(leaf) == 1
        assert ".claude/.credentials.json" in credentials
        assert {".claude/settings.json", ".claude/settings.local.json"} <= set(settings)


# ── Claude's own store under a profile's directory ───────────────────────────


class TestStore:
    def test_claude_config_dir_follows_the_walk(self, home, signed_in):
        from kiro_crew.acp.claude_settings_sources import claude_config_dir

        assert claude_config_dir() == home / ".claude"
        seen = []
        assert dirs.every_config_dir(lambda: seen.append(claude_config_dir()) or True)() is True
        assert seen == [home / ".claude", Path(signed_in)]
        assert claude_config_dir() == home / ".claude"

    def test_a_walk_outranks_the_gateways_own_override(self, home, signed_in, monkeypatch):
        from kiro_crew.acp.claude_settings_sources import claude_config_dir

        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / "relocated"))
        seen = []
        dirs.every_config_dir(lambda: seen.append(claude_config_dir()) or True)()
        assert seen == [home / "relocated", Path(signed_in)]

    def test_a_profiled_session_store_is_found_and_deleted(self, home, signed_in, tmp_path):
        from kiro_crew.agent_sdk import claude_transcripts as ct

        cwd = tmp_path / "work"
        cwd.mkdir()
        project = signed_in / "projects" / ct.claude_project_dir_name(str(cwd))
        (project / SID / "subagents").mkdir(parents=True)
        (project / f"{SID}.jsonl").write_text("{}\n")
        history = signed_in / "file-history" / SID
        history.mkdir(parents=True)
        (history / "a@v1").write_text("x")

        assert ct.claude_session_dir(SID, str(cwd)) == project / SID
        assert ct.delete_claude_session_files(SID, str(cwd)) is True
        assert not (project / f"{SID}.jsonl").exists()
        assert not (project / SID).exists()
        assert not history.exists()
        # Nothing holds it now, so the answer is the default store's again.
        assert ct.claude_session_dir(SID, str(cwd)).parent.parent.parent == home / ".claude"

    def test_a_default_session_store_is_still_found_and_deleted(self, home, signed_in, tmp_path):
        from kiro_crew.agent_sdk import claude_transcripts as ct

        cwd = tmp_path / "work"
        cwd.mkdir()
        project = home / ".claude" / "projects" / ct.claude_project_dir_name(str(cwd))
        (project / SID).mkdir(parents=True)
        (project / f"{SID}.jsonl").write_text("{}\n")
        assert ct.claude_session_dir(SID, str(cwd)) == project / SID
        assert ct.delete_claude_session_files(SID, str(cwd)) is True
        assert not (project / f"{SID}.jsonl").exists() and not (project / SID).exists()

    def test_no_profile_directory_changes_nothing(self, home, tmp_path):
        from kiro_crew.agent_sdk import claude_transcripts as ct

        cwd = tmp_path / "work"
        cwd.mkdir()
        expected = home / ".claude" / "projects" / ct.claude_project_dir_name(str(cwd)) / SID
        assert ct.claude_session_dir(SID, str(cwd)) == expected
        assert ct.delete_claude_session_files(SID, str(cwd)) is True
        assert ct.delete_claude_session_files("not-a-session-id", str(cwd)) is False
        assert ct.claude_session_dir("not-a-session-id", str(cwd)) is None

    def test_the_importers_hold_the_wrapped_functions(self):
        from kiro_crew.acp import claude_background_calls
        from kiro_crew.agent_sdk import claude_transcripts as ct
        from kiro_crew.providers import acp_lifecycle

        assert claude_background_calls.claude_session_dir is ct.claude_session_dir
        assert acp_lifecycle.delete_claude_session_files is ct.delete_claude_session_files
        assert ct.delete_claude_session_files.__wrapped__.__name__ == "delete_claude_session_files"
        assert ct.claude_session_dir.__wrapped__.__name__ == "claude_session_dir"

    def test_the_seed_reads_the_profiles_own_user_settings(self, home, signed_in, tmp_path):
        from kiro_crew.acp.claude_settings_sources import other_sources_define_available_models

        work = tmp_path / "work"
        work.mkdir()
        (home / ".claude").mkdir()
        (home / ".claude" / "settings.json").write_text('{"availableModels": ["local-model"]}')
        managed = tmp_path / "managed"
        default_client = SimpleNamespace(_extra_env={})
        cloud_client = SimpleNamespace(_extra_env={"CLAUDE_CONFIG_DIR": str(signed_in)})
        assert other_sources_define_available_models(
            work, config_dir=cp.session_config_dir(default_client), managed_dir=managed
        )
        assert not other_sources_define_available_models(
            work, config_dir=cp.session_config_dir(cloud_client), managed_dir=managed
        )


# ── seam canaries: the upstream names these hooks depend on ──────────────────


class TestUpstreamSeams:
    def test_user_settings_lookup_takes_a_config_dir(self):
        from kiro_crew.acp.claude_settings_sources import other_sources_define_available_models

        params = inspect.signature(other_sources_define_available_models).parameters
        assert "config_dir" in params and "managed_dir" in params

    def test_the_client_keeps_its_extra_env(self):
        from kiro_crew.acp import client_class

        assert "self._extra_env = extra_env or {}" in inspect.getsource(client_class)

    def test_the_security_lists_still_splice_both_tables(self):
        # The fences reach ``security.paths_lists`` through these two calls; the
        # security package itself is not edited (it is held to a line budget).
        source = (Path(cp.__file__).resolve().parents[1] / "security" / "paths_lists.py").read_text(
            encoding="utf-8"
        )
        assert "*host_auth.credential_leaves()," in source
        assert "native_allow_stores.home_leaves()" in source

    def test_call_sites_are_present(self):
        root = Path(cp.__file__).resolve().parents[1]
        for rel, needle in [
            ("acp/claude_settings_sources.py", "walked_config_dir()"),
            ("acp/client_claude_seed.py", "config_dir=session_config_dir(self)"),
            ("agent_sdk/claude_transcripts.py", "every_config_dir(delete_claude_session_files)"),
            ("agent_sdk/claude_transcripts.py", "first_existing_config_dir(claude_session_dir)"),
            ("agent_sdk/host_auth.py", "import CREDENTIAL_LEAVES as profile_leaves"),
            ("agent_sdk/native_allow_stores.py", "import SETTINGS_LEAVES as profile_leaves"),
        ]:
            assert needle in (root / rel).read_text(encoding="utf-8"), rel
