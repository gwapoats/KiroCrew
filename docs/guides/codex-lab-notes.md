# Running KiroCrew on the Codex backend: lab notes

Notes from a lab instance (v0.10.0 wheel, codex-acp 2.2.2, a Bedrock-served model) that
reached parity with the Claude backend for shell, chat orchestration, cron, lessons,
subagents and skills. This branch is for experiments; the two `fix(codex)` commits
underneath it are the upstream candidates.

## Second instance next to a running one

| Variable | Why |
|---|---|
| `KIROCREW_HOME=$LAB/home`, `KIRO_HOME=$LAB/home/kiro` | Own data and agent specs, so the running instance's specs are never rewritten |
| `KIROCREW_PORT=5490`, `--no-tunnel` | Own loopback port, nothing published |
| `KIROCREW_PROFILE=standalone` | The public wheel refuses an enterprise-looking host without the companion package |
| `CODEX_HOME=$LAB/codex-home` | A Codex config with no MCP servers or plugins of its own |
| `CODEX_PATH`, `CODEX_ACP_BIN` | The Codex binary that knows your model provider, and the adapter |
| `AWS_CONFIG_FILE`, `AWS_SHARED_CREDENTIALS_FILE` | Point outside `~/.aws`, which Crew masks from Codex children |
| `KIROCREW_LAB_CODEX_OWN_SECRET=1` | Opt into the lab-only commit on this branch (see below) |

Bedrock credentials: refresh a `credential_process` JSON cache outside the sandbox and
serve it with `credential_process = /bin/cat $LAB/aws/creds.json`, so the SDK re-reads it
when the previous credentials expire.

## `$CODEX_HOME/AGENTS.md`

```markdown
# Running inside KiroCrew

## Shell commands
KiroCrew already runs you inside its own OS sandbox, and Codex's own sandbox cannot
start inside it: a sandboxed command fails with `bwrap: Failed to make / slave` before it
runs. Request escalated permissions for every shell command
(`sandbox_permissions: "require_escalated"` with a one-line justification); KiroCrew's
permission gate then decides.

## KiroCrew tools come first
- Subagents: `spawn_run` / `spawn_sub_agents`, which the dashboard shows.
- New chats, messages to other chats, folders: the kirocrew-dashboard tools.
- Schedules: kirocrew-cron. Lessons and memory: `learn_add`, `memory_recall`.
- Skills: `skill_search`, then `skill_fetch`.
```

Codex 0.164 cannot switch off its native `collaboration.*` subagent tools
(`features.multi_agent = false` and `agents.max_depth = 0` were both tried); the
instruction above is what steers a plain "use subagents" request to `spawn_sub_agents`.

## The lab-only commit

Crew's MCP servers run as Codex's children, and Crew masks `.local_secret` and `run/`
from every Codex child, so every gateway-backed Crew tool fails with "the gateway
received no internal credential". The commit drops only the active `KIROCREW_HOME`'s two
paths from the Codex mask, never the default `~/.kiro/crew`'s, behind
`KIROCREW_LAB_CODEX_OWN_SECRET=1`. A first attempt that moved the leaf names to the
child-readable list also unmasked the default home's credential, because those lists
apply to every data home. The proper fix is to move those readers behind the gateway.
