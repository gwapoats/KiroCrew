"""LOCAL PATCH (not upstream): run chosen dashboard chats on a second Claude Code profile.

Configured in the owner-only ``~/.kiro/crew/config.json`` (write-protected for the
agent's file tools, read-only inside the sandbox)::

    "local_ext": {"claude_profiles": {"agents": {"anthropic": "cloud"}}}

Keys are agent TEMPLATE names (the agent a chat runs); values name a profile in
:data:`~kiro_crew.local_ext.claude_profile_dirs.PROFILE_DIRS` (``cloud`` is
``~/.claude-cloud``).

A profile is a Claude Code config directory of its own: its own sign-in (on macOS
a Keychain entry of its own, because Claude Code names the entry after the
directory), its own ``settings.json``, its own transcripts and auto-memory. So one
gateway can keep ``~/.claude`` pointed at a local model and never signed in, and
run the chats of one template on a Claude subscription signed in under the profile.

Rules:

* The profile only supplies the ``extra_env`` allocation argument, naming
  ``CLAUDE_CONFIG_DIR``. That is the plumbing ``aws.project_profiles`` already
  rides: the warm pool is bypassed (``bypass_env``), the session record keeps the
  map, and ``allocation_identity`` hands it to a reset or compaction successor.
  The adapter process is started with the variable, so claude-agent-acp and the
  ``claude`` it runs read the same directory.
* Same chats as ``local_ext.chat_backends`` routes: an ordinary dashboard chat
  running a TEMPLATE. Background calls, sub-agents, crons and Bots never take a
  profile, so they stay on ``~/.claude``.
* Refused before any session is claimed (:class:`ClaudeProfileUnavailable`), with
  an error row in the chat, when the stored name is no known profile, the chat
  would not run on Claude Code, the profile's directory does not exist (nobody
  signed in there), or the gateway's own environment or ``agent.env`` carries a
  variable that would pick the provider or the credential over the profile's
  sign-in. Nothing runs on ``~/.claude`` in its place.
* A live session started under another config directory than the chat should use
  is reset before the claim (``chat_backends.reset_if_mismatched``), so the turn
  cold-starts under the right one and Crew replays the transcript into it.
* With no ``local_ext.claude_profiles`` object in the config, every function here
  answers "nothing to do".
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from kiro_crew.crewmate_harness import CrewmateHarnessUnavailable
from kiro_crew.local_ext.claude_profile_dirs import CONFIG_DIR_ENV, PROFILE_DIRS, config_dir

logger = logging.getLogger(__name__)

SECTION = "local_ext"
FEATURE = "claude_profiles"
CODE = "claude_profile_unavailable"

#: Variables that pick Claude Code's provider or credential ahead of the sign-in
#: stored in its config directory. One of these inherited by the adapter would run
#: a profiled chat somewhere other than the profile's own account, so a profile
#: refuses to start while the gateway's environment or ``agent.env`` sets one.
PROVIDER_ENV_NAMES: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_VERTEX",
)


class ClaudeProfileUnavailable(CrewmateHarnessUnavailable):
    """A template's Claude profile cannot be used for this turn (fail closed).

    A :class:`~kiro_crew.crewmate_harness.CrewmateHarnessUnavailable`, so every
    caller that already refuses an unstartable Bot pin or backend route handles
    this one the same way. Never echoes the raw stored value.
    """

    code = CODE

    def __init__(self, template: str, reason: str, fix: str) -> None:
        from kiro_crew.agent_sdk.backends import ACP_BACKEND_CLAUDE, backend_to_stored

        self.crew = ""
        self.stored = backend_to_stored(ACP_BACKEND_CLAUDE)
        self.template = template
        self.reason = reason
        RuntimeError.__init__(
            self,
            f"{CODE}: chats on agent {template!r} use a Claude profile "
            f"(local_ext.claude_profiles in ~/.kiro/crew/config.json), and {reason}. "
            f"{fix} Nothing runs on the default Claude config in its place.",
        )


# ── configuration ────────────────────────────────────────────────────────────


def feature_section(cfg: Any) -> dict | None:
    """The ``local_ext.claude_profiles`` object, or ``None`` when not configured."""
    extra = getattr(cfg, "_extra_sections", None)
    section = extra.get(SECTION) if isinstance(extra, dict) else None
    feature = section.get(FEATURE) if isinstance(section, dict) else None
    return feature if isinstance(feature, dict) else None


def agent_profiles(cfg: Any) -> dict | None:
    """The template -> profile map; ``{}`` when configured with no map, ``None`` when off."""
    feature = feature_section(cfg)
    if feature is None:
        return None
    agents = feature.get("agents")
    return agents if isinstance(agents, dict) else {}


def profile_for_template(cfg: Any, template: str | None) -> str | None:
    """The profile *template*'s chats run under, or ``None``.

    ``None``: not configured, no entry, or an empty value (inherit ``~/.claude``).
    A value that names no profile raises :class:`ClaudeProfileUnavailable`.
    """
    profiles = agent_profiles(cfg)
    if not profiles or not template or template not in profiles:
        return None
    stored = profiles[template]
    if isinstance(stored, str) and not stored.strip():
        return None
    name = stored.strip() if isinstance(stored, str) else None
    if name not in PROFILE_DIRS:
        known = ", ".join(sorted(PROFILE_DIRS))
        raise ClaudeProfileUnavailable(
            template,
            "the stored value names no profile this build knows",
            f"Set it to one of: {known}; or remove the entry.",
        )
    return name


def profile_env(profile: str) -> dict[str, str]:
    """The ``extra_env`` a session of *profile* is allocated with."""
    return {CONFIG_DIR_ENV: config_dir(profile)}


# ── whether a profile can start ──────────────────────────────────────────────


def inherited_provider_env(cfg: Any, environ: Any = None) -> tuple[str, ...]:
    """The :data:`PROVIDER_ENV_NAMES` the adapter would inherit, sorted. Names only.

    From the gateway's own environment and from ``agent.env``, the two layers a
    spawn applies beneath a session's ``extra_env``.
    """
    environ = os.environ if environ is None else environ
    agent_env = getattr(getattr(cfg, "agent", None), "env", None)
    agent_env = agent_env if isinstance(agent_env, dict) else {}
    return tuple(
        sorted(name for name in PROVIDER_ENV_NAMES if environ.get(name) or agent_env.get(name))
    )


async def ensure_startable(cfg: Any, template: str, profile: str, backend: str) -> None:
    """Raise :class:`ClaudeProfileUnavailable` unless *profile* can serve this turn.

    *backend* is the runtime backend the chat will run on (its route, else its
    default). Checked in the order a person would fix them.
    """
    from kiro_crew.agent_sdk.backends import ACP_BACKEND_CLAUDE

    if backend != ACP_BACKEND_CLAUDE:
        raise ClaudeProfileUnavailable(
            template,
            "this chat would not run on Claude Code",
            f"Route the agent to it (local_ext.chat_backends.agents: {template!r}: "
            '"claude"), or remove its profile.',
        )
    inherited = inherited_provider_env(cfg)
    if inherited:
        raise ClaudeProfileUnavailable(
            template,
            f"the gateway's environment or agent.env sets {', '.join(inherited)}, which "
            "would override the profile's own sign-in",
            "Take them out of the gateway's environment and restart it; a local model's "
            "endpoint and key belong in the env block of ~/.claude/settings.json, which only "
            "the default config reads.",
        )
    directory = config_dir(profile)
    if not await asyncio.to_thread(os.path.isdir, directory):
        raise ClaudeProfileUnavailable(
            template,
            f"its config directory ~/{PROFILE_DIRS[profile]} does not exist",
            f'Sign in once from your own terminal: {CONFIG_DIR_ENV}="$HOME/'
            f'{PROFILE_DIRS[profile]}" claude',
        )


# ── the live session ─────────────────────────────────────────────────────────


def stamped_config_dir(sessions: Any, session_key: str) -> str | None:
    """The ``CLAUDE_CONFIG_DIR`` the live session on *session_key* was allocated with."""
    table = getattr(sessions, "_sessions", None)
    fold = getattr(sessions, "_fold_key", None)
    if not isinstance(table, dict) or not callable(fold):
        return None
    env = getattr(table.get(fold(session_key)), "extra_env", None)
    value = env.get(CONFIG_DIR_ENV) if isinstance(env, dict) else None
    return value if isinstance(value, str) and value else None


def session_config_dir(client: Any) -> Path | None:
    """The profile config directory an ACP client's process was started with, or ``None``.

    For the one upstream call that resolves Claude's user settings for a session
    (``client_claude_seed``): ``None`` keeps its own ``~/.claude`` resolution.
    """
    env = getattr(client, "_extra_env", None)
    value = env.get(CONFIG_DIR_ENV) if isinstance(env, dict) else None
    return Path(value) if isinstance(value, str) and value else None
