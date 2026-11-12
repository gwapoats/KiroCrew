"""LOCAL PATCH (not upstream): where a Claude profile's own config directory lives.

Standard library only. ``security.paths_lists`` reads the leaf tables while the
security package is still loading, and ``agent_sdk.claude_transcripts`` wraps two
functions with the walkers below, so this module must not reach the config loader,
the ACP layer or anything else of Crew's.

A profile is a second Claude Code config directory beside ``~/.claude``. The table
is fixed rather than free-form because the path fences in ``security.paths_lists``
are static lists of ``$HOME``-relative leaves: a directory named here is fenced the
way ``~/.claude`` is, and one that is not named here cannot be a profile. To add a
profile, add its row; the fences, the profile lookup and the walkers all follow.
"""

from __future__ import annotations

import contextvars
import functools
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

#: The variable Claude Code and claude-agent-acp both read their config directory from.
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"

#: Profile name -> its config directory, ``$HOME``-relative.
PROFILE_DIRS: dict[str, str] = {
    "cloud": ".claude-cloud",
}

#: Claude Code's OAuth token file (Linux and Windows; macOS keeps it in the
#: Keychain). Read+write fenced, as ``.claude/.credentials.json`` is.
CREDENTIAL_LEAVES: tuple[str, ...] = tuple(
    f"{directory}/.credentials.json" for directory in PROFILE_DIRS.values()
)

#: The settings files a session of the profile takes allow rules, hooks and its
#: permission mode from. Write fenced, as their ``~/.claude`` twins are.
SETTINGS_LEAVES: tuple[str, ...] = tuple(
    f"{directory}/{name}"
    for directory in PROFILE_DIRS.values()
    for name in ("settings.json", "settings.local.json")
)


def config_dir(profile: str) -> str:
    """The absolute config directory of *profile*, spelled without a trailing separator.

    The spelling matters: on macOS Claude Code files its login in the Keychain
    under a name derived from this exact string, so the directory a session is
    handed must be spelled as the terminal sign-in spelled it
    (``CLAUDE_CONFIG_DIR="$HOME/.claude-cloud" claude``). Raises ``KeyError`` for
    a name the table does not carry.
    """
    return os.path.join(os.path.expanduser("~"), *PROFILE_DIRS[profile].split("/"))


def existing_config_dirs() -> tuple[str, ...]:
    """The profile config directories present on disk. Blocking (one stat each)."""
    found = []
    for profile in PROFILE_DIRS:
        directory = config_dir(profile)
        try:
            if os.path.isdir(directory):
                found.append(directory)
        except OSError:
            continue
    return tuple(found)


# ── walking a profile's directory with code written for ``~/.claude`` ────────

_walked: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kiro_crew_local_ext_claude_walked_config_dir", default=None
)


def walked_config_dir() -> Path | None:
    """The profile directory the current call is walking, or ``None`` (the usual case).

    Read by ``acp.claude_settings_sources.claude_config_dir``, so a function that
    resolves Claude's config directory through it can be run once more per profile.
    """
    value = _walked.get()
    return Path(value) if value else None


def _run_in(directory: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    token = _walked.set(directory)
    try:
        return fn(*args, **kwargs)
    finally:
        _walked.reset(token)


def every_config_dir(fn: _F) -> _F:
    """Run *fn* for ``~/.claude``, then again for each profile directory on disk.

    For a cleanup that answers whether it completed: the answer is true only when
    every run's was. A session id is a UUID, so a run in a directory that never
    held the session deletes nothing.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> bool:
        done = bool(fn(*args, **kwargs))
        for directory in existing_config_dirs():
            done = bool(_run_in(directory, fn, args, kwargs)) and done
        return done

    return wrapper  # type: ignore[return-value]


def first_existing_config_dir(fn: _F) -> _F:
    """*fn*'s path for ``~/.claude`` when it exists, else the first profile's that does.

    For a lookup that spells a path under Claude's config directory. Nothing
    existing anywhere answers what *fn* answered unwrapped.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        found = fn(*args, **kwargs)
        if found is None or _exists(found):
            return found
        for directory in existing_config_dirs():
            other = _run_in(directory, fn, args, kwargs)
            if other is not None and _exists(other):
                return other
        return found

    return wrapper  # type: ignore[return-value]


def _exists(path: Any) -> bool:
    try:
        return os.path.exists(path)
    except (OSError, TypeError, ValueError):
        return False
