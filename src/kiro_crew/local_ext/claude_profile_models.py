"""LOCAL PATCH (not upstream): keep each Claude profile's model list to itself.

Every Claude Code session advertises the models its config directory is served:
a ``~/.claude`` session pointed at a local model lists that model, a
``~/.claude-cloud`` session signed in to a subscription lists the subscription's.
Upstream reads the picker's list from the NEWEST ``claude_code`` session, and its
cold snapshot from one ``claude_code`` cache key, so the two profiles overwrite
each other: a chat on the cloud profile shows the local list until its own first
turn, and a local chat shows the cloud list after one.

Here the list is kept per profile:

* A live session's profile is read off the ``CLAUDE_CONFIG_DIR`` it was started
  with (``extra_env``); no variable is the default profile, ``~/.claude``.
* Each advertised list is also recorded under ``claude_code@<profile>``
  (``claude_code@default`` for ``~/.claude``) beside upstream's own key, which is
  left as it was for every other reader (the settings seed, the wire-id fold).
* A chat whose template runs on a profile gets ``models_backend`` =
  ``claude@<profile>`` from selection-capabilities; ``GET /api/models`` splits it
  and answers with that profile's sessions, then its snapshot.
* A chat on ``~/.claude`` reads only default-profile sessions and snapshot.

A model is also kept to the profile that serves it. Claude Code lists a model it
was handed but does not know as a row named by its id, so a session started on
another profile's model advertises that model back and it lands in this profile's
list. A profile's OWN models are therefore the ids it advertised under a name of
their own (:func:`native_ids`), and a model that is another profile's own and not
this one's:

* is not handed to a session of this profile: the session starts on its own
  default (:func:`client_model`, read where the ACP client is built), and a
  session already running keeps its model (:func:`foreign_to_session`, read by
  ``model_scope.pin_applies``). The chat's stored pick is left as it is.
* is not listed in this profile's picker (:func:`own_rows`).

Inert when no profile directory exists on disk and no session carries a profile:
nothing is recorded and every filter passes every session. The model rules are
inert while only one profile has advertised a list.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence

from kiro_crew.local_ext.claude_profile_dirs import (
    CONFIG_DIR_ENV,
    PROFILE_DIRS,
    config_dir,
    existing_config_dirs,
)

logger = logging.getLogger(__name__)

#: Joins a backend or namespace to a profile name: ``claude@cloud``, ``claude_code@cloud``.
SEP = "@"
#: The profile name of ``~/.claude`` in snapshot keys.
DEFAULT_PROFILE = "default"
#: The registry namespace Claude Code's ids live in.
NAMESPACE = "claude_code"
#: The "no pin" model: the backend picks.
AUTO = "auto"


# ── which profile a session runs on ──────────────────────────────────────────


def _env_of(owner: Any) -> dict | None:
    for holder in (owner, getattr(owner, "_runtime", None)):
        env = getattr(holder, "_extra_env", None)
        if isinstance(env, dict):
            return env
    return None


def profile_of_dir(directory: str | None) -> str | None:
    """The profile whose config directory is *directory*, ``None`` for none or an unknown one."""
    if not directory:
        return None
    wanted = os.path.normpath(os.path.expanduser(directory))
    for name in PROFILE_DIRS:
        if os.path.normpath(config_dir(name)) == wanted:
            return name
    return None


def owner_profile(owner: Any) -> str | None:
    """The profile an ACP client or session handle was started on; ``None`` is ``~/.claude``."""
    env = _env_of(owner)
    value = env.get(CONFIG_DIR_ENV) if env else None
    return profile_of_dir(value) if isinstance(value, str) else None


def provider_profile(provider: Any) -> str | None:
    """:func:`owner_profile` of a provider's client (or of the provider itself)."""
    client = getattr(provider, "client", None)
    return owner_profile(client if client is not None else provider)


def matches(provider: Any, profile: str | None) -> bool:
    """Whether *provider* runs on *profile* (``None`` = ``~/.claude``)."""
    try:
        return provider_profile(provider) == profile
    except Exception:  # noqa: BLE001 - a display filter; never fail the list
        return profile is None


# ── the per-profile snapshot ─────────────────────────────────────────────────


def snapshot_key(namespace: str, profile: str | None) -> str:
    """The cache key of *profile*'s list in *namespace*."""
    return f"{namespace}{SEP}{profile or DEFAULT_PROFILE}"


def record_snapshot(
    owner: Any, namespace: str, ids: Sequence[str], rows: Sequence[object] | None
) -> bool:
    """Also file a session's advertised list under its profile's key. ``True`` when it changed.

    Only for ``claude_code``, and only once a profile is in use: the session
    carries one, or a profile directory exists. Never raises.
    """
    if namespace != "claude_code" or not ids:
        return False
    try:
        profile = owner_profile(owner)
        if profile is None and not existing_config_dirs():
            return False
        from kiro_crew import model_registry

        return model_registry.refresh_advertised_models(
            snapshot_key(namespace, profile), ids, rows=rows, confirm_shrink=True
        )
    except Exception:  # noqa: BLE001 - best effort beside upstream's own write
        logger.debug("local_ext.claude_profile_models: snapshot not recorded", exc_info=True)
        return False


def snapshot_rows(namespace: str, profile: str | None) -> list[dict[str, str]] | None:
    """*profile*'s snapshot as picker rows, or ``None`` when it has none yet."""
    from kiro_crew import model_registry

    key = snapshot_key(namespace, profile)
    if not model_registry.advertised_models(key):
        return None
    return own_rows(model_registry.advertised_model_rows(key), profile, namespace)


# ── which models are a profile's own ─────────────────────────────────────────


def _profiles() -> list[str | None]:
    """Every profile, ``None`` (``~/.claude``) first."""
    return [None, *PROFILE_DIRS]


def _key(model_id: str) -> str:
    """*model_id* folded like the registry folds an id to match it against a served list."""
    from kiro_crew import model_registry

    return model_registry._normalize_advertised_key(model_id) or model_id


def _listed(model_id: str, ids: Iterable[str]) -> bool:
    """Whether *model_id* is one of *ids*, in any spelling of the same id."""
    ids = list(ids)
    if model_id in ids:
        return True
    wanted = _key(model_id)
    return any(_key(other) == wanted for other in ids)


def native_ids(profile: str | None) -> frozenset[str]:
    """The ids *profile*'s sessions advertised under a name of their own.

    A row named by its id is how Claude Code lists a model it was handed and does
    not know, so such a row says what the session was given, not what the profile
    serves. Empty until the profile has advertised a named list.
    """
    from kiro_crew import model_registry

    rows = model_registry.advertised_model_rows(snapshot_key(NAMESPACE, profile))
    return frozenset(
        row["model_name"]
        for row in rows
        if row.get("display_name") and row["display_name"] != row["model_name"]
    )


def _own_elsewhere(model_id: str, profile: str | None) -> str | None:
    """A profile other than *profile* that serves *model_id* as its own, by snapshot name."""
    for other in _profiles():
        if other != profile and _listed(model_id, native_ids(other)):
            return other or DEFAULT_PROFILE
    return None


def foreign_to(model_id: str, profile: str | None) -> str | None:
    """The profile *model_id* belongs to when that is not *profile*, else ``None``.

    ``None`` for a model *profile* serves itself and for one no profile claims.
    A profile that has advertised no named list yet serves nothing known, so a
    model another profile serves is that profile's until this one lists it: one
    session on its own default costs less than a session whose every turn asks
    for a model the account does not have.
    """
    if not model_id or _listed(model_id, native_ids(profile)):
        return None
    return _own_elsewhere(model_id, profile)


def client_model(client: Any) -> str:
    """The model a new ACP client starts on: its own, unless another profile serves it.

    Read where the client is built, before its settings seed and its startup model
    are written, so the id never reaches Claude Code. ``auto`` then leaves the
    session on its config directory's own default. Never raises.
    """
    model = getattr(client, "_model", "") or ""
    try:
        from kiro_crew.agent_sdk.backends import ACP_BACKEND_CLAUDE

        if model in ("", AUTO) or getattr(client, "backend", "") != ACP_BACKEND_CLAUDE:
            return model
        profile = owner_profile(client)
        owner = foreign_to(model, profile)
    except Exception:  # noqa: BLE001 - keep the model the caller asked for
        logger.debug("local_ext.claude_profile_models: client model not scoped", exc_info=True)
        return model
    if owner is None:
        return model
    logger.info(
        "local_ext.claude_profiles: model %s is the %s profile's, not the %s profile's; "
        "session %s starts on its own default",
        model,
        owner,
        profile or DEFAULT_PROFILE,
        getattr(client, "_session_key", "") or "?",
    )
    return AUTO


def foreign_to_session(model_id: str, namespace: str, advertised: Sequence[str] | None) -> bool:
    """``model_scope.pin_applies``: a named profile's own model this session does not list.

    For a session that is already running (a claimed warm process, a re-applied
    pin), judged by the list the session itself advertised. Only a model that is
    some named profile's own and not ``~/.claude``'s is refused, so an install
    without profiles, and every id both profiles serve, is untouched. Never raises.
    """
    if namespace != NAMESPACE or not model_id or not advertised:
        return False
    try:
        if _listed(model_id, advertised) or _listed(model_id, native_ids(None)):
            return False
        return any(_listed(model_id, native_ids(name)) for name in PROFILE_DIRS)
    except Exception:  # noqa: BLE001 - upstream's own answer stands
        logger.debug("local_ext.claude_profile_models: pin not scoped", exc_info=True)
        return False


def own_rows(
    rows: Sequence[dict[str, str]], profile: str | None, namespace: str = NAMESPACE
) -> list[dict[str, str]]:
    """Picker *rows* of *profile* without the rows that echo another profile's model.

    Dropped: a row named by its id (see :func:`native_ids`) whose model another
    profile serves as its own. A model no profile claims stays (a custom model
    the user set in this profile). Only Claude Code's rows are read. Never raises.
    """
    if namespace != NAMESPACE:
        return list(rows)
    try:
        return [
            row
            for row in rows
            if (row.get("display_name") or row.get("model_name")) != row.get("model_name")
            or _own_elsewhere(row.get("model_name", ""), profile) is None
        ]
    except Exception:  # noqa: BLE001 - a display filter; never fail the list
        logger.debug("local_ext.claude_profile_models: rows not filtered", exc_info=True)
        return list(rows)


# ── the picker's ``models_backend`` ──────────────────────────────────────────


def split_models_backend(value: str | None) -> tuple[str | None, str | None]:
    """``claude@cloud`` -> ``("claude", "cloud")``; a value without a known profile is kept whole."""
    if not value or SEP not in value:
        return value, None
    backend, _, profile = value.partition(SEP)
    if profile not in PROFILE_DIRS:
        return value, None
    return backend, profile


def models_backend_for(backend_stored: str, profile: str | None) -> str | None:
    """The ``models_backend`` a profiled chat's pickers read, ``None`` without a profile."""
    return f"{backend_stored}{SEP}{profile}" if profile else None


def query_without_profile(query: Any) -> tuple[Any, str | None]:
    """``GET /api/models``'s query with ``backend=claude@cloud`` split into ``claude`` + profile."""
    raw = query.get("backend") if hasattr(query, "get") else None
    backend, profile = split_models_backend(raw)
    if profile is None:
        return query, None
    items: Iterable[tuple[str, str]] = query.items()
    rewritten = {k: (backend if k == "backend" else v) for k, v in items}
    return rewritten, profile


async def profile_models_field(
    state: Any, slot: Any, field: dict[str, str], config: Any = None
) -> dict[str, str]:
    """selection-capabilities' ``models_backend`` for *slot*, profile-aware.

    *field* is upstream's answer, kept unless the chat's next turn runs on a
    profile; then ``models_backend`` is ``claude@<profile>``. A profile only ever
    runs on Claude Code (``claude_profiles.ensure_startable``), so that holds even
    while a session on another harness is still live. Never raises.
    """
    import asyncio

    from kiro_crew.agent_sdk.backends import ACP_BACKEND_CLAUDE, backend_to_stored
    from kiro_crew.dashboard.chat_utils import effective_session_key

    try:
        if config is None:
            from kiro_crew.config import KiroCrewConfig

            config = await asyncio.to_thread(KiroCrewConfig.load)
        profile = await slot_profile(state, slot, effective_session_key(slot), config)
    except Exception:  # noqa: BLE001 - a display read; keep upstream's answer
        logger.debug("local_ext: profile models field unresolved", exc_info=True)
        return field
    value = models_backend_for(backend_to_stored(ACP_BACKEND_CLAUDE), profile)
    return field if value is None else {"models_backend": value}


async def slot_profile(state: Any, slot: Any, session_key: str, cfg: Any) -> str | None:
    """The profile a chat's pickers should list: the one its NEXT turn runs on.

    Resolved from the chat's template like the turn resolves it, not read off a
    live session: a chat just switched from a local agent still holds its local
    session, which the next turn resets (``chat_backends.reset_if_mismatched``).
    ``None`` for ``~/.claude``, for a chat no profile can apply to, and whenever
    profiles are not configured.
    """
    from kiro_crew.local_ext.claude_profiles import agent_profiles

    if cfg is None or not agent_profiles(cfg):
        return None
    import asyncio

    from kiro_crew.local_ext.chat_backends import _routable

    if not await _routable(state, slot, session_key):
        return None
    return await asyncio.to_thread(_cold_slot_profile, cfg, slot, session_key)


def _cold_slot_profile(cfg: Any, slot: Any, session_key: str) -> str | None:
    """Blocking: the profile a cold slot's next session takes, resolved like its route."""
    from kiro_crew.config.loader import resolve_agent_bindings
    from kiro_crew.local_ext.chat_backends import execution_is_template
    from kiro_crew.local_ext.claude_profiles import ClaudeProfileUnavailable, profile_for_template
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
        return None
    execution = getattr(bindings, "execution_context", None)
    if not execution_is_template(execution, getattr(bindings, "selection_kind", "")):
        return None
    try:
        return profile_for_template(
            cfg, getattr(bindings, "kiro_agent", "") or getattr(slot, "agent", "") or None
        )
    except ClaudeProfileUnavailable:
        return None
