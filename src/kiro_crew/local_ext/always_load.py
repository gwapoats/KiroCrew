"""LOCAL PATCH (not upstream): have Claude Code load chosen MCP tools up front.

Claude Code defers MCP tools behind its ToolSearch loader and adds a tool's schema
to the request's ``tools`` array the first time the model loads it. A
prefix-caching local server (oMLX) renders ``tools`` at the top of the prompt, so
each such load re-prefills the whole conversation. A tool whose ``tools/list``
entry carries ``_meta["anthropic/alwaysLoad"] = true`` is sent from the first
request instead and never changes the prompt head later. That is Claude Code's
own per-tool switch; its per-server ``alwaysLoad`` is the same switch, but
claude-agent-acp does not forward it from ACP's ``mcpServers``.

Configured in ``~/.kiro/crew/config.json``::

    "local_ext": {"always_load": {"tools": [
        "mcp__kirocrew-core__memory_recall",
        "mcp__kirocrew-core__learn_*",
        "mcp__searxng"
    ]}}

An entry is the name Claude Code gives the model, ``mcp__<server>__<tool>``, an
``fnmatch`` glob over it, or ``mcp__<server>`` for every tool of that server.

Applied where Crew answers or relays ``tools/list``: its own stdio servers
(``mcp_shared_dispatch``) and the gateway's relay for pooled servers
(``mcp_gateway.backend_mcp_apps``). Not covered: Claude Code's built-in tools
(WebFetch, WebSearch, ...) and any server Claude Code starts itself rather than
through the gateway. Other harnesses ignore the key. With no
``local_ext.always_load`` section every function here leaves listings as they are.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

SECTION = "local_ext"
FEATURE = "always_load"
META_KEY = "anthropic/alwaysLoad"

#: Seconds a loaded pattern list is reused, so a burst of sessions listing tools
#: does not re-read config.json for each one, while an edit still lands quickly.
_CACHE_TTL = 5.0
_cache_lock = threading.Lock()
_cache: tuple[float, tuple[str, ...]] | None = None


def patterns(cfg: Any) -> tuple[str, ...]:
    """The configured ``local_ext.always_load.tools`` entries; ``()`` when off."""
    extra = getattr(cfg, "_extra_sections", None)
    section = extra.get(SECTION) if isinstance(extra, dict) else None
    feature = section.get(FEATURE) if isinstance(section, dict) else None
    raw = feature.get("tools") if isinstance(feature, dict) else None
    if not isinstance(raw, list):
        return ()
    return tuple(p.strip() for p in raw if isinstance(p, str) and p.strip())


def _load_patterns() -> tuple[str, ...]:
    """Patterns from the current config file; ``()`` if it cannot be read."""
    global _cache  # noqa: PLW0603
    now = time.monotonic()
    with _cache_lock:
        if _cache is not None and now - _cache[0] < _CACHE_TTL:
            return _cache[1]
    try:
        from kiro_crew.config import KiroCrewConfig

        loaded = patterns(KiroCrewConfig.load())
    except Exception:  # an unreadable config pins nothing, as with no section
        logger.debug("local_ext.always_load: config not readable; pinning nothing", exc_info=True)
        loaded = ()
    with _cache_lock:
        _cache = (now, loaded)
    return loaded


def matches(qualified: str, pats: tuple[str, ...]) -> bool:
    """Whether ``mcp__<server>__<tool>`` is named by one of *pats*."""
    return any(
        fnmatch.fnmatchcase(qualified, p) or fnmatch.fnmatchcase(qualified, p + "__*") for p in pats
    )


def pin_tools(server: str, tools: Any, pats: tuple[str, ...]) -> Any:
    """*tools* with the matching entries marked ``alwaysLoad``.

    Marked entries are copies, so a listing the server caches is never mutated.
    An entry whose ``_meta`` is present but not an object is left alone: that
    container belongs to the server, and the gateway's visibility check already
    treats it as unreadable. Returns *tools* itself when nothing is configured.
    """
    if not pats or not isinstance(tools, list) or not isinstance(server, str) or not server:
        return tools
    out = []
    for tool in tools:
        name = tool.get("name") if isinstance(tool, dict) else None
        meta = tool.get("_meta") if isinstance(tool, dict) else None
        if (
            isinstance(name, str)
            and (meta is None or isinstance(meta, dict))
            and matches(f"mcp__{server}__{name}", pats)
        ):
            tool = {**tool, "_meta": {**(meta or {}), META_KEY: True}}
        out.append(tool)
    return out


def pin_listing(server: str, tools: Any) -> Any:
    """:func:`pin_tools` with the patterns from config (stdio ``tools/list`` answer)."""
    return pin_tools(server, tools, _load_patterns())


def pin_result(server: str, result: Any) -> None:
    """Mark a relayed ``tools/list`` *result* in place (gateway relay)."""
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        result["tools"] = pin_listing(server, result["tools"])
