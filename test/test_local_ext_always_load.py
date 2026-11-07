"""LOCAL PATCH: ``kiro_crew.local_ext.always_load`` -- tools Claude Code loads up front.

Unit tests of the marking, plus one test per call site through the production
code (a real stdio server process; the gateway's tools/list relay), so a rebase
that moves either seam fails here rather than silently pinning nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiro_crew import mcp_shared, platform_compat
from kiro_crew.local_ext import always_load as al
from kiro_crew.mcp_gateway import backend_mcp_apps

SRC = str(Path(mcp_shared.__file__).resolve().parents[1])
KEY = "anthropic/alwaysLoad"


def _tool(name, **extra):
    return {"name": name, "description": name, "inputSchema": {"type": "object"}, **extra}


def _cfg(tools):
    return SimpleNamespace(_extra_sections={"local_ext": {"always_load": {"tools": tools}}})


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(al, "_cache", None)


def test_patterns_reads_the_section_and_drops_junk():
    assert al.patterns(_cfg([" mcp__a__x ", "", 3, None, "mcp__b"])) == ("mcp__a__x", "mcp__b")
    assert al.patterns(SimpleNamespace(_extra_sections={})) == ()
    assert (
        al.patterns(
            SimpleNamespace(_extra_sections={"local_ext": {"always_load": {"tools": "mcp__a"}}})
        )
        == ()
    )
    assert al.patterns(SimpleNamespace()) == ()


def test_exact_glob_and_whole_server_entries():
    tools = [_tool("memory_recall"), _tool("learn_add"), _tool("learn_list"), _tool("browser")]
    out = al.pin_tools(
        "kirocrew-core", tools, ("mcp__kirocrew-core__memory_recall", "mcp__kirocrew-core__learn_*")
    )
    assert [KEY in (t.get("_meta") or {}) for t in out] == [True, True, True, False]
    whole = al.pin_tools("searxng", [_tool("web_search"), _tool("web_url_read")], ("mcp__searxng",))
    assert all(t["_meta"][KEY] is True for t in whole)
    # A server-level entry names only that server, not one sharing its prefix.
    assert "_meta" not in al.pin_tools("searxng2", [_tool("web_search")], ("mcp__searxng",))[0]


def test_existing_meta_is_kept_and_inputs_are_not_mutated():
    ui = {"ui": {"visibility": ["model", "app"], "resourceUri": "ui://x"}}
    original = _tool("x", _meta=ui)
    out = al.pin_tools("s", [original], ("mcp__s__x",))
    assert out[0]["_meta"] == {**ui, KEY: True}
    assert original["_meta"] == ui and KEY not in original["_meta"]
    assert out[0] is not original


def test_unreadable_meta_unconfigured_and_odd_inputs_are_left_alone():
    bad = _tool("x", _meta="not-an-object")
    assert al.pin_tools("s", [bad], ("mcp__s__x",)) == [bad]
    tools = [_tool("x")]
    assert al.pin_tools("s", tools, ()) is tools
    assert al.pin_tools("", tools, ("mcp__s__x",)) is tools
    assert al.pin_tools("s", None, ("mcp__s__x",)) is None
    assert al.pin_tools("s", ["junk", _tool("x")], ("mcp__s__x",))[0] == "junk"


def test_unreadable_config_pins_nothing(monkeypatch):
    from kiro_crew.config import KiroCrewConfig

    def boom():
        raise RuntimeError("bad config")

    monkeypatch.setattr(KiroCrewConfig, "load", staticmethod(boom))
    tools = [_tool("x")]
    assert al.pin_listing("s", tools) is tools


SERVER = textwrap.dedent("""
    from kiro_crew import mcp_shared

    def list_tools():
        return [{"name": n, "description": n, "inputSchema": {"type": "object"}} for n in ("keep", "other")]

    mcp_shared.run_mcp_stdio_loop("pin-test", "1.0", list_tools, lambda name, args: "ok")
    """)


def _list_from_server(tmp_path: Path, config: dict | None) -> list[dict]:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    if config is not None:
        (home / "config.json").write_text(json.dumps(config))
        os.chmod(home / "config.json", 0o600)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "KIROCREW_HOME": str(home),
        "PYTHONPATH": SRC,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    proc = subprocess.run(
        [sys.executable, "-c", SERVER],
        input=request.encode(),
        capture_output=True,
        env=env,
        cwd=str(tmp_path),
        timeout=120,
    )
    lines = [json.loads(line) for line in proc.stdout.decode().splitlines() if line.strip()]
    (reply,) = [m for m in lines if m.get("id") == 1]
    return reply["result"]["tools"]


@pytest.mark.skipif(not platform_compat.IS_POSIX, reason="stdio harness is POSIX-only")
def test_stdio_server_marks_configured_tools(tmp_path):
    tools = _list_from_server(
        tmp_path, {"local_ext": {"always_load": {"tools": ["mcp__pin-test__keep"]}}}
    )
    by_name = {t["name"]: t for t in tools}
    assert by_name["keep"]["_meta"][KEY] is True
    assert "_meta" not in by_name["other"]


@pytest.mark.skipif(not platform_compat.IS_POSIX, reason="stdio harness is POSIX-only")
def test_stdio_server_without_the_section_lists_tools_unchanged(tmp_path):
    tools = _list_from_server(tmp_path, None)
    assert all("_meta" not in t for t in tools)


def _relay(listing: dict, *, stub: str = "stub-1") -> dict:
    backend = SimpleNamespace(
        pool_key=SimpleNamespace(server_name="searxng"),
        _served_tool_surfaces={},
        _apps_declared_uris={},
        _audit_visibility_withhold=lambda pending, hidden: None,
    )
    pending = SimpleNamespace(stub_uuid=stub, method="tools/list", list_paginated=False)
    msg = {"jsonrpc": "2.0", "id": 7, "result": listing}
    delivered_later = asyncio.run(
        backend_mcp_apps._maybe_intercept_ui_result(backend, pending, msg)
    )
    assert delivered_later is False  # the caller still delivers the listing itself
    return msg["result"]


def test_gateway_relay_marks_configured_tools(monkeypatch):
    monkeypatch.setattr(al, "_load_patterns", lambda: ("mcp__searxng__web_search",))
    hidden_only = _tool("app_only", _meta={"ui": {"visibility": ["app"]}})
    result = _relay({"tools": [_tool("web_search"), _tool("web_url_read"), hidden_only]})
    names = {t["name"]: t for t in result["tools"]}
    assert names["web_search"]["_meta"][KEY] is True
    assert "_meta" not in names["web_url_read"]
    assert "app_only" not in names  # upstream's visibility strip still runs first


def test_gateway_relay_leaves_internal_probes_alone(monkeypatch):
    monkeypatch.setattr(al, "_load_patterns", lambda: ("mcp__searxng",))
    prefix = backend_mcp_apps.INTERNAL_STUB_PREFIXES[0]
    result = _relay({"tools": [_tool("web_search")]}, stub=prefix + "probe")
    assert "_meta" not in result["tools"][0]
