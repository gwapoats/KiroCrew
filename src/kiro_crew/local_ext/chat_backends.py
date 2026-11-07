"""LOCAL PATCH (not upstream): route ordinary dashboard chats to a backend by agent template.

Configured in the owner-only ``~/.kiro/crew/config.json`` (write-protected for the
agent's file tools, read-only inside the sandbox)::

    "local_ext": {"chat_backends": {"agents": {"codex-coder": "codex"}}}

Keys are agent TEMPLATE names (the agent a chat runs); values use the
``agent.acp_backend`` spelling (``kiro``, ``kas``, ``claude``, ``codex``, ...).

Rules (the ones upstream's per-chat backend review settled on):

* Precedence stays ``spawn_run(backend=)`` > this route > a Bot's harness pin >
  the member route > ``agent.acp_backend``. The route only supplies the
  ``backend_override`` the one gate (``members.select_provider_backend``) already
  takes, so it rides the ``spawn_run`` plumbing: warm-pool bypass, the session's
  stamp, ``allocation_identity`` for reset/compaction successors, per-backend model
  scoping and the provider-switch transcript replay.
* Only ordinary dashboard chats running a TEMPLATE. Never a Bot's own chat (DM),
  a Bot's worker, any member selection, an app slot, a channel-linked or mirrored
  chat, or a session bound to a remote Session Environment.
* A route naming a harness this gateway cannot start refuses the turn with an
  error row in the chat (:class:`ChatBackendUnavailable`); nothing falls back to
  the default backend.
* A live session on another backend than the chat should run on is reset before
  the claim, so the turn cold-starts on the right harness (Crew replays the
  transcript into it). For a chat with no route this happens only to a session
  this route started, so an unrouted chat keeps upstream behaviour exactly.
* With no ``local_ext.chat_backends`` object in the config, every function here
  answers "nothing to do".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable

logger = logging.getLogger(__name__)

SECTION = "local_ext"
FEATURE = "chat_backends"
CODE = "chat_backend_unavailable"


class ChatBackendUnavailable(CrewmateHarnessUnavailable):
    """A route names a harness this gateway cannot start (fail closed).

    A :class:`~kiro_crew.crewmate_harness.CrewmateHarnessUnavailable`, so every
    caller that already refuses an unstartable Bot pin (the turn's error path,
    the selection-capabilities 409, the eager-spawn refusal predicate) handles
    this one the same way. Never echoes the raw stored value.
    """

    code = CODE

    def __init__(self, template: str, stored: object) -> None:
        from kiro_crew.crewmate_harness import harness_label

        self.crew = ""
        self.stored = stored if isinstance(stored, str) else ""
        self.template = template
        label = harness_label(stored) if isinstance(stored, str) else "an unknown harness"
        RuntimeError.__init__(
            self,
            f"{CODE}: chats on agent {template!r} are routed to {label} "
            "(local_ext.chat_backends in ~/.kiro/crew/config.json), which this gateway "
            "cannot start. Fix or remove that route, or pick another agent for this "
            "chat. Nothing runs on another harness in its place.",
        )


# ── configuration ────────────────────────────────────────────────────────────


def feature_section(cfg: Any) -> dict | None:
    """The ``local_ext.chat_backends`` object, or ``None`` when not configured."""
    extra = getattr(cfg, "_extra_sections", None)
    section = extra.get(SECTION) if isinstance(extra, dict) else None
    feature = section.get(FEATURE) if isinstance(section, dict) else None
    return feature if isinstance(feature, dict) else None


def agent_routes(cfg: Any) -> dict | None:
    """The template -> backend map; ``{}`` when configured with no map, ``None`` when off."""
    feature = feature_section(cfg)
    if feature is None:
        return None
    agents = feature.get("agents")
    return agents if isinstance(agents, dict) else {}


def route_for_template(cfg: Any, template: str | None) -> str | None:
    """The runtime backend id *template*'s chats are routed to, or ``None``.

    ``None``: not configured, no entry, or an empty value (inherit). A value this
    gateway cannot start raises :class:`ChatBackendUnavailable`. Read as a stored
    ``agent.acp_backend`` is: ``kiro`` moves onto KAS where the loader moves it.
    """
    routes = agent_routes(cfg)
    if not routes or not template or template not in routes:
        return None
    stored = routes[template]
    if isinstance(stored, str) and not stored.strip():
        return None
    if not isinstance(stored, str):
        raise ChatBackendUnavailable(template, stored)
    from kiro_crew.crewmate_harness import CrewBackendPin

    kas_bundle = bool(getattr(getattr(cfg, "agent", None), "kas_bundle", False))
    try:
        return CrewBackendPin(template, stored.strip(), kas_bundle).resolve()
    except CrewmateHarnessUnavailable as exc:
        raise ChatBackendUnavailable(template, stored.strip()) from exc


def default_backend(cfg: Any, session_key: str) -> str:
    """The backend *session_key* gets with no route, from the one selection gate."""
    from kiro_crew.crewmate_harness import crewmate_backend_pin
    from kiro_crew.members import select_provider_backend

    return select_provider_backend(
        session_key,
        cfg.agent.member_acp_backend,
        cfg.agent.acp_backend,
        crew_backend=crewmate_backend_pin(cfg, session_key),
    )


# ── which chats a route may apply to ─────────────────────────────────────────


def slot_is_routable(state: Any, slot: Any, session_key: str) -> bool:
    """Whether *slot* is an ordinary dashboard chat (in-memory checks only)."""
    from kiro_crew.members import (
        DM_SLOT_MODE,
        is_crewmate_worker_session,
        is_member_session_key,
    )

    if is_member_session_key(session_key) or is_crewmate_worker_session(session_key):
        return False
    if getattr(slot, "mode", "") == DM_SLOT_MODE:
        return False
    for attr in (
        "_app",
        "app_agent_owner",
        "linked_session_key",
        "_slack_channel",
        "_slack_linked",
        "channel_origin",
        "_channel_runtime_origin",
    ):
        if getattr(slot, attr, None):
            return False
    sessions = getattr(state, "sessions", None) if state is not None else None
    get_mirror = getattr(sessions, "get_mirror_link", None)
    if callable(get_mirror) and get_mirror(session_key) is not None:
        return False
    return True


def execution_is_template(execution: Any, selection_kind: str = "") -> bool:
    """Whether the chat runs a template as no member (a Bot worker keeps its member id)."""
    if execution is None:
        return selection_kind == "template"
    return getattr(execution, "selection_kind", "") == "template" and not getattr(
        execution, "member_id", None
    )


async def _runs_remotely(session_key: str) -> bool:
    from kiro_crew.session_environment import session_environments_enabled, session_runs_remotely

    if not session_environments_enabled():
        return False
    return await asyncio.to_thread(session_runs_remotely, session_key)


async def _routable(state: Any, slot: Any, session_key: str) -> bool:
    return slot_is_routable(state, slot, session_key) and not await _runs_remotely(session_key)


# ── the live session ─────────────────────────────────────────────────────────


def live_backend(provider: Any) -> str | None:
    """The backend a live provider's own capabilities name, or ``None``."""
    from kiro_crew.agent_sdk.capabilities import SessionCapabilities

    caps = getattr(provider, "capabilities", None)
    return caps.backend if isinstance(caps, SessionCapabilities) else None


def stamped_override(sessions: Any, session_key: str) -> str | None:
    """The ``backend_override`` the live session on *session_key* was started with."""
    table = getattr(sessions, "_sessions", None)
    fold = getattr(sessions, "_fold_key", None)
    if not isinstance(table, dict) or not callable(fold):
        return None
    value = getattr(table.get(fold(session_key)), "backend_override", None)
    return value if isinstance(value, str) else None


def reset_if_mismatched(
    state: Any, slot: Any, session_key: str, route: str | None, default: str
) -> bool:
    """Arm the turn's pre-claim reset when the live session runs the wrong backend.

    The consume right before the claim (``_consume_pending_reset``) performs it, so
    the turn cold-starts on the right harness. Returns whether it armed.
    """
    sessions = getattr(state, "sessions", None)
    provider = sessions.get_provider(session_key) if sessions is not None else None
    live = live_backend(provider) if provider is not None else None
    if live is None:
        return False
    if route is not None:
        wrong = live != route
    else:
        # No route: only a session a route started is ours to move back.
        wrong = stamped_override(sessions, session_key) is not None and live != default
    if not wrong:
        return False
    from kiro_crew.dashboard.session_control import arm_model_switch_reset

    logger.info(
        "local_ext.chat_backends: slot %s live session runs %r, should run %r; resetting",
        getattr(slot, "key", "?"),
        live,
        route if route is not None else default,
    )
    arm_model_switch_reset(state, slot)
    return True


# ── call sites ───────────────────────────────────────────────────────────────


async def turn_route(
    state: Any,
    slot: Any,
    session_key: str,
    cfg: Any,
    agent: str | None,
    execution_context: Any,
) -> dict[str, str]:
    """Dashboard turn admission: the extra allocation kwargs for this turn.

    ``{}`` (no route, or the route names the backend the chat gets anyway, so the
    warm pool stays usable) or ``{"backend_override": <id>}``. Also arms the reset
    of a live session on the wrong backend. Raises :class:`ChatBackendUnavailable`
    for a route that cannot start, before any session is claimed.
    """
    if cfg is None or agent_routes(cfg) is None:
        return {}
    if not execution_is_template(execution_context) or not await _routable(
        state, slot, session_key
    ):
        return {}
    route = route_for_template(cfg, agent or getattr(slot, "agent", "") or None)
    default = default_backend(cfg, session_key)
    reset_if_mismatched(state, slot, session_key, route, default)
    if route is None or route == default:
        return {}
    return {"backend_override": route}


async def blocks_prewarm(state: Any, slot: Any, session_key: str, cfg: Any, bindings: Any) -> bool:
    """Eager spawn: whether a routed chat must be left to its first real turn.

    The prefetch allocates with no route, so it would register a session on the
    default backend. A broken route also blocks (the turn reports it).
    """
    if cfg is None or not agent_routes(cfg):
        return False
    execution = getattr(bindings, "execution_context", None)
    if not execution_is_template(execution, getattr(bindings, "selection_kind", "")):
        return False
    if not await _routable(state, slot, session_key):
        return False
    try:
        route = route_for_template(cfg, getattr(bindings, "kiro_agent", "") or slot.agent)
    except ChatBackendUnavailable:
        return True
    blocked = route is not None and route != default_backend(cfg, session_key)
    if blocked:
        logger.info("Eager spawn: %s has a backend route; left to first turn", slot.key)
    return blocked


def _cold_slot_route(cfg: Any, slot: Any, session_key: str) -> str | None:
    """Blocking: the route a cold slot's next session takes, resolved like a turn."""
    from kiro_crew.config.loader import resolve_agent_bindings
    from kiro_crew.session_agent_selection import resolve_session_agent_bindings

    try:
        bindings = resolve_session_agent_bindings(
            resolve_agent_bindings,
            cfg,
            session_key,
            getattr(slot, "agent", "") or None,
            getattr(slot, "project", "") or None,
            agent_kind=getattr(slot, "agent_kind", "") or "",
        )
    except Exception:  # noqa: BLE001 - a display read; the turn reports real errors
        logger.debug("local_ext.chat_backends: cold slot %s unresolved", session_key, exc_info=True)
        return None
    execution = getattr(bindings, "execution_context", None)
    if not execution_is_template(execution, getattr(bindings, "selection_kind", "")):
        return None
    route = route_for_template(cfg, bindings.kiro_agent or getattr(slot, "agent", "") or None)
    return None if route is None or route == default_backend(cfg, session_key) else route


async def slot_backend_override(cfg: Any, slot: Any, session_key: str) -> str | None:
    """A cold slot's ``backend_override`` (model picker / effort reads), or ``None``.

    Raises :class:`ChatBackendUnavailable` for a route that cannot start, which the
    callers already answer as they answer an unstartable Bot pin.
    """
    if cfg is None or not agent_routes(cfg) or not await _routable(None, slot, session_key):
        return None
    return await asyncio.to_thread(_cold_slot_route, cfg, slot, session_key)


async def catalog_session_backend(state: Any, cfg: Any, session_key: str) -> str | None:
    """``GET /api/agents/catalog``'s ``session_backend`` for a routed chat, else ``None``.

    The config spelling of the harness the chat's CURRENT agent runs on. ``None``
    leaves the upstream answer in place (no route, no chat, or a broken route).
    """
    if not session_key or cfg is None or not agent_routes(cfg) or state is None:
        return None
    slot = getattr(state, "_slots", {}).get(session_key.split(":", 1)[-1])
    if slot is None:
        return None
    from kiro_crew.acp_backends import backend_to_stored
    from kiro_crew.dashboard.chat_utils import effective_session_key

    key = effective_session_key(slot)
    provider = state.sessions.get_provider(key) if getattr(state, "sessions", None) else None
    live = live_backend(provider) if provider is not None else None
    if live is not None:
        return backend_to_stored(live)
    try:
        route = await slot_backend_override(cfg, slot, key)
    except ChatBackendUnavailable:
        return None
    return backend_to_stored(route) if route is not None else None
