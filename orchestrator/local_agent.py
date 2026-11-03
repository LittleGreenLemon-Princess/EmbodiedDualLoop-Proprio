"""Explicitly non-isolated Claude pilot; the Docker launcher stays the default."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from orchestrator.files import write_json


def provider_env(settings: Path) -> dict[str, str]:
    """Load provider routing only, excluding hooks, plugins and user permissions."""
    data = json.loads(settings.read_text())
    allowed = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL", "ANTHROPIC_REASONING_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL", "API_TIMEOUT_MS",
    }
    return {k: str(v) for k, v in data.get("env", {}).items() if k in allowed}


def command(binary: str, model: str, phase: str, effort: str = "", *,
            role='development', handoff=None) -> list[str]:
    prompt = (
        "This is an explicitly NON-ISOLATED local end-to-end pilot, not a formal benchmark. "
        "Work only in your current workspace. Do not inspect parent directories, host "
        "files, simulator sources, credentials, other episodes or private ledgers. "
        "Treat instruction.md, scripts/, primitives/ and observations/ as read-only. "
        + ("Treat skills/ and lessons/ as read-only frozen knowledge. " if phase == "test" else "")
        + "Read instruction.md and primitives/primitives.md. Work autonomously toward "
        "the phase goal using python scripts/env.py. Native-resolution camera PNGs "
        "are in observations/; inspect at most two current_cam_*.png frames per iteration. "
        "Never alter raw feedback or task checks. Record reusable skills and lessons "
        "during Playground. When finished or unable to continue, call "
        "python scripts/env.py finish. Do not launch background work."
    )
    if role == 'repair':
        prompt = (
            'You are System 2, invoked ONLY because the fixed native-task program stopped. '
            'This is a NON-ISOLATED development run. Work only in the current workspace. '
            'Do not inspect parents, other scenes, host files, simulator sources or credentials. '
            'Read instruction.md, primitives/primitives.md and ' + str(handoff) + '. '
            'Reuse returned evidence; inspect at most two current camera PNGs per iteration. '
            'Keep skills/, lessons/, scripts/, primitives/, instruction.md and observations/ '
            'unchanged. Continue from the current physical state; do not redo completed motion. '
            'Use python scripts/env.py exec ONLY for zero-action observation and geometry. '
            'You MUST NOT move, call delegate, reset or finish; the host controller owns these. '
            'Diagnose the raw failure and propose ONE local code/geometry/sensing repair in '
            'submission/. Copy handoff.proposal_template into the UTF-8 JSON file at '
            'handoff.proposal_path and follow handoff.proposal_contract. The outer task '
            'object and task.entry are required; diagnosis or code alone is not a proposal. '
            'The template contains the current source paths, parameters and limits. Copy '
            'changed sources to submission/ and update task.entry/includes to those files. '
            'Omitted task fields inherit the current revision; an explicit parameters '
            'object replaces all parameters, so keep every required parameter. Omit '
            'selector to inherit the complete frozen selector; do not change its kind, '
            'model or other settings. Use the actual base_limits; do not enlarge budgets or lower '
            'confidence. The repaired code still has skill_init/context/options/tick, zero '
            'actions in init/context/options, and at most one action in tick. System 1 owns '
            'repeated execution. Do not coach the next direction/step or split repeated '
            'closing into one-step task restarts. Changing only task names, budgets, '
            'counters or revisions is not a repair. Keep the native goal and physical '
            'evidence standard. If no evidenced repair is available, write '
            '{"stop":true,"reason":"..."} to proposal_path. After writing it, exit '
            'without calling finish. Do not background work.'
        )
    return [
        binary, "--print", "--model", model,
        *(["--effort", effort] if effort else []),
        "--permission-mode", "acceptEdits",
        "--tools", "Bash,Read,Write,Edit,Glob,Grep",
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep",
        "--disable-slash-commands", "--no-session-persistence",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "", "--output-format", "stream-json", "--verbose", prompt,
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="", choices=["", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--phase", choices=["playground", "test"], required=True)
    parser.add_argument('--role', choices=['development', 'repair'], default='development')
    parser.add_argument('--handoff')
    parser.add_argument('--log-dir', type=Path)
    args = parser.parse_args(argv)
    episode = args.episode.resolve()
    logs = args.log_dir or episode / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        binary = (os.environ.get("EMBODIEDEVO_CLAUDE_BIN") or shutil.which("claude")
                  or str(Path.home() / ".local/bin/claude"))
        settings = Path(os.environ.get(
            "EMBODIEDEVO_CLAUDE_SETTINGS", str(Path.home() / ".claude/settings.json")
        ))
        routing = provider_env(settings)
        args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        env = {k: v for k, v in os.environ.items() if k in {
            "LANG", "LC_ALL", "TZ", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
            "https_proxy", "http_proxy", "all_proxy", "no_proxy",
        }}
        env.update(routing)
        env.update({
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath
                    + ":/usr/local/bin",
            "HOME": str(args.home.resolve()),
            "CLAUDE_CONFIG_DIR": str(args.home.resolve() / ".claude"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "DISABLE_AUTOUPDATER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "EMBODIEDEVO_WORKSPACE": str(episode / "workspace"),
            "EMBODIEDEVO_SOCKET": str(args.socket.resolve()),
            # A workspace nested below the host repository must not discover
            # its parent's Git history through an innocent `git log`.
            "GIT_CEILING_DIRECTORIES": str((episode / 'workspace').resolve().parent),
        })
        secrets = [v for k, v in routing.items() if "KEY" in k or "TOKEN" in k]
        write_json(logs / "launch-config.json", {
            "model": args.model, "reasoning_effort": args.effort or "provider-default",
            "agent_isolation": "none",
            'role': args.role,
        })
        with (logs / "session.jsonl").open("w") as transcript, \
                (logs / "stderr.txt").open("w") as errors:
            process = subprocess.Popen(
                command(binary, args.model, args.phase, args.effort,
                        role=args.role, handoff=args.handoff), cwd=episode / "workspace",
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=errors, text=True,
            )
            for line in process.stdout:
                for secret in secrets:
                    if secret:
                        line = line.replace(secret, "<redacted>")
                transcript.write(line)
                transcript.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "result":
                    write_json(logs / "usage.json", {
                        "source": "claude-result", "available": True,
                        "usage": event.get("usage"), "model_usage": event.get("modelUsage"),
                        "total_cost_usd": event.get("total_cost_usd"),
                    })
            return process.wait()
    except Exception as exc:
        write_json(logs / "launch-error.json", {
            "type": type(exc).__name__, "message": str(exc),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
