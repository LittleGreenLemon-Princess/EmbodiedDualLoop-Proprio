# EmbodiedEvo development boundaries

## Dual-loop experiments

Read `docs/DUAL_LOOP.zh-CN.md` before using the experimental `delegate` operation.
System 2 owns the ordered task plan, code generation and repair. Delegate one
parameterized subtask to the host System 1 runtime, then review its report and
fresh physical evidence. Continue the next ready subtask, or fix only the failed
part and delegate a new revision. Do not treat code completion, an EE target or
a selector's judgement as verified object manipulation.

Label this experimental condition `dual-loop-v1`; the original reproduction and
its source/skill snapshots remain separately identified. Do not tune against
archived Test programs and then report those same scenes as held-out evaluation.

## Fixed task program condition

`dual_loop_frozen_task` adds the explicitly separate condition
`fixed-base-with-local-system2-repair-v1`. In Playground, export the complete
parameterized task under `skills/task_program.json` and referenced `skills/`
sources, and actually delegate that code/parameters before freezing. The host
seals its hash. Normal Test execution runs the fixed program through Jev without
starting System 2; only a returned failure/threshold/evidence gap can start a
bounded local repair. System 2 may inspect with zero-action exec and write a
proposal, but the host owns delegate/finish. Never transfer Test repairs to the
next scene or modify the frozen base. Report primary and recovered native success,
interventions, revisions and repair cost separately. This is not strict frozen-code
evaluation and must not be merged with the original skills-only transfer results.

OpenRouter is an explicit optional provider adaptation, documented in
`docs/OPENROUTER.zh-CN.md`. Its host API plus offline shell-container path is
labeled `openrouter-only-dual-loop-adaptation-v1`, not a native Jev reproduction.
Keep existing provider defaults and supplied helpers unchanged when adding
provider-specific transport handling. Never retry a robot/session/tool operation
because a model API response was incomplete.

## Original reproduction boundaries

This repository contains the RoboDojo reproduction source. Read README.md and
docs/PROTOCOL.md before changing execution or evaluation behavior.

- Preserve `workspace-session-v1`: persistent Python sessions and exactly four
  primitives: `get_instruction()`, `get_observation()`, `step(action)`, and
  Playground-only `reset()`. Keep the primitive manual handwritten.
- Only `skills/` and `lessons/` transfer from Playground to Test. Freeze them
  before selecting Test scenes. Each Test gets a fresh agent, workspace and
  simulator; knowledge is read-only, and reset and cross-scene learning are forbidden.
- Preserve task goals, native action limits, success checks and raw feedback.
  Do not add a model to summarize observations or videos.
- Keep ledgers, private rewards, scene plans and simulator sources outside the
  agent workspace. The agent image has no simulator or GPU access.
- Treat `vendor/` as read-only after provisioning. Put compatibility adapters
  in `simulators/` or `integrations/`; do not redirect to another project checkout.
- Every invocation needs an explicit `--run-dir` with a `YYYYMMDDHHMM` prefix.
  Never copy results, credentials, environments or machine configuration into Git.
- Preserve failed attempts. Model-written code failures count as model failures.
  Only confirmed infrastructure failures are candidates for an explicitly recorded
  rerun; crashes, timeouts and missing reports alone do not establish causation.
- Use `python -m unittest discover -s tests -v` for CPU validation. GPU experiments,
  container builds and model calls are separate operations requiring actual resources.
- Keep the third-party recipe text, license and checksums unchanged. Record its use
  as supplied task knowledge, separately from learned Skills/Lessons.
