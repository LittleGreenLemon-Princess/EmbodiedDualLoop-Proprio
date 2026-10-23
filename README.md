# EmbodiedDualLoop

This derivative adds opt-in public proprioceptive feedback and a versioned,
hash-locked skill library inspired by ASPIRE. See
[本体反馈与技能库融合](docs/PROPRIOCEPTION.zh-CN.md) for design, provenance and usage.
The new experiment condition is `dual-loop-proprio-v1`; baseline inputs are unchanged by default.

EmbodiedEvo reproduction plus an experimental second runtime for execution decisions.

System 2 plans the workflow, writes parameterized skills, and repairs failures. It delegates
one subtask to System 1. System 1 repeatedly observes, selects a legal option, executes a
bounded action, records its measured effects, and returns a report when finished or stuck.
The selector can be a deterministic baseline or TypeSafe Jev; robot control stays local.

## Start here

- [双循环怎么运行、怎么接入抓碗](docs/DUAL_LOOP.zh-CN.md)
- [固定基础任务程序与按需 System 2 修复](docs/FIXED_TASK.zh-CN.md)
- [Original setup and reproduction guide](README.embodiedevo.md)
- [Original execution/evaluation protocol](docs/PROTOCOL.md)
- `SERVER_SNAPSHOT.json`: hashes and origins of the server source backup.
- `skill_snapshots/`: four frozen skill/lesson snapshots, preserved as supplied or learned.
- `program_snapshots/`: generated programs and execution source history from the server.

## CPU demonstration

Python 3.11+ and the normal host dependencies are required:

```sh
python -m pip install -r requirements-host.txt -e .
python -m scripts.smoke_dual_loop --run-dir runs/202610191530_dual_loop_cpu
python -m scripts.smoke_dual_loop --run-dir runs/202610191531_dual_loop_stuck --stuck
python -m unittest discover -s tests -v
```

Use a fresh YYYYMMDDHHMM run directory each time. This demonstration uses an artificial
waypoint follower: no model calls, no robot physics, no grasp verification. The normal case
stops at `awaiting_verification`; the stuck case requests System 2 help.

## Runtime source

| Path | Role |
| --- | --- |
| `orchestrator/dual_loop/` | System 1 loop, task descriptions, selectors, reports and evidence |
| `integrations/rim_pick_loop.py` | Parameterized, incremental rim-pick state machine |
| `integrations/skills/thin_rim_grasp/stepwise_pick.py` | Four skill entry points |
| `orchestrator/session.py` | System 2 delegates through the existing session |
| `orchestrator/session_client.py` | The `python scripts/env.py delegate ...` client |
| `tests/test_dual_loop.py` | CPU checks of limits, handoffs, evidence and selectors |

The original four robot primitives and native task budgets remain in force. The new path is
labeled `dual-loop-v1`; it is an experimental condition, not an unchanged baseline.
Jev integration and control-flow checks do not establish robot success or real-time latency.
See the guide for current sensing and verification limits.

Original provenance and recipe attribution remain in [docs/SOURCE.md](docs/SOURCE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new project-wide license is assigned here.
