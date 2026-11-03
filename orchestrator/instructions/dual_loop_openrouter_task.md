# Develop once; execute the fixed task program with OpenRouter System 1 on new scenes

Phase: {{PHASE}}. Condition: openrouter-only-dual-loop-adaptation-v1, under the
experimental dual-loop-v1 runtime. Read primitives/primitives.md and the current
unchanged native instruction. There are no mandatory perturbation diagnostics.
Native task ID: {{NATIVE_TASK_ID}}.

## Playground: your deliverable is a COMPLETE parameterized task program

You are the real System 2. Develop a reusable program for THIS native task, not
just the next motor step or a transcript of this one scene. Maintain a plan in
scratch/plan.json and implement the entire task state machine, current-object
perception, parameter options, failure/progress detection and physical evidence.

Put the complete entry in skills/task_program.py and its manifest in
skills/task_program.json EARLY, then debug these same exported files. The entry
defines skill_init(parameters), skill_context(state), skill_options(state),
skill_tick(state, option). Include supplied skills/ helpers where appropriate.
Initialize object locations, identities and current progress from fresh public
observations in skill_init, never from saved development coordinates or Python
variables. Keep task constants/geometry assumptions/bounds as explicit parameters.
Read get_instruction() at each new scene: target colors/objects may change while
the native task ID stays the same. The host binds OpenRouter System 1's task goal to that current
public instruction, rather than reusing the Playground object description.
Use runtime state and observations to decide which stage is already complete.
The frozen program must start in a fresh interpreter/scene without Playground
globals, scratch files, submission files or external Python imports.

Expose meaningful bounded physical parameter adjustments at the stages where
they can help: for example signed XYZ/attitude target offsets, variable correction
size, or a bounded contact-geometry parameter. Do not claim parameter adaptation
if the whole menu is only advance/escalate. Parameters need not change on every
successful run; preserve what OpenRouter System 1 actually selected. Every option must accurately
describe its code effect, including translation, quaternion rotation and gripper
behavior. Provide current measured errors/object evidence and recent effects;
do not hide an action's capabilities behind an opaque label or prescribe the
next answer. These are code capabilities, not motor hints from System 2.

Each context/options/init call has ZERO actions. Each tick has at most ONE
native step(action). OpenRouter System 1 owns repeated observation, parameter selection and
execution inside the whole task. You own workflow, code, sensing and debugging.
Keep the other arm/gripper at its measured state. Do not schedule each primitive
from System 2, coach the next signed direction/step, or restart one-action tasks
to avoid low confidence. Repeated close/settle/hold belongs in the same loop.

In this fixed-program profile the host adds context.measured_grippers from
current public observations: both measured openings, per-action opening delta,
closing/opening/stationary motion and consecutive stationary action counts.
RoboDojo ARX uses 1=open, 0=closed. These are measurements, not commands.
Include commanded gripper state and measured closure progress in your own
stage feedback too. A stationary EE during jaw closure is expected: do not
diagnose a stall from EE motion alone. Bound close/settle actions and use
measured jaw convergence together with physical object evidence; jaw stoppage
or a partially closed gripper alone never establishes contact or a grasp.
When repairing after closure, inspect current grasp/occlusion evidence and
resume the justified stage instead of blindly reopening or re-detecting a new
target. Missing measurements mean unknown, not zero or successful closure.

This run explicitly enables `proprioception-v1`. The host may add measured
end-effector pose, arm-joint positions, normalized gripper opening, validity
flags, and measured before/after action deltas from the existing public
observation state. Use only present, valid fields and retain unknown values as
unknown. This interface does not expose force, torque, contact force, or tactile
measurements; do not infer or claim those signals.

ALL movement is delegated with the OpenRouter System 1 selector. System 1 and System 2 both use the same pinned OpenRouter DeepSeek model in this explicitly noncanonical adaptation condition; this is not a Jev run. Ordinary exec is restricted to
zero-action observation/math. No reset, alternate connection or direct motor loop.
Use the existing native action/execution limits: {{NATIVE_ACTION_LIMIT}} actions,
{{EXECUTION_BUDGET}} execution requests. Do not change native goals or success.
Use min_confidence >= .5, API timeout 120 seconds, and state age 120 seconds for this host-routed model condition. Rotation
error target * inverse(measured) is the REQUIRED world-axis correction: apply its
sign directly, do not negate it. Factorized correction is for free space only.

## Export format

Read the actual get_instruction() and identify the current native task. Write:

```json
{
  "schema": "frozen-task-v1",
  "native_task": "{{NATIVE_TASK_ID}}",
    "observation_profile": "proprioception-v1",
  "max_system2_interventions": 2,
  "selector": {"kind": "openrouter", "model": "deepseek/deepseek-v4.1-flash", "effort": "high", "timeout": 120},
  "task": {
    "task_id": "complete-native-task",
    "revision": 1,
    "goal": "THE_FULL_UNCHANGED_NATIVE_GOAL",
    "entry": "skills/task_program.py",
    "includes": ["skills/ee_pose_servo.py"],
    "parameters": {},
    "max_decisions": 300,
    "max_action_calls": 300,
    "max_seconds": 400,
    "max_state_age_seconds": 120,
    "min_confidence": 0.5
  }
}
```

Replace task ID/goal, parameters/includes and suitable bounded budgets with the
actual task. ALL referenced files must live inside skills/ and be real files.
You may include extra helper functions in the entry or in other skills/ files.
Do not add new primitives or read host configuration or other scenes.

Run the exact exported program BEFORE finishing:

```sh
python -c 'import json; p=json.load(open("skills/task_program.json")); json.dump(p["task"],open("submission/development_task.json","w"))'
python scripts/env.py delegate submission/development_task.json --selector openrouter --model deepseek/deepseek-v4.1-flash --effort high --timeout 120 --observation-profile proprioception-v1
```

env.py expects the TASK object, not the outer package. The commands above copy
manifest["task"] for development execution; select the OpenRouter selector from the manifest. Update both after a repair. The outer package is
for the host. A Bash timeout of 600000 ms supports
the synchronous delegation. Inspect raw reports and current camera/RGB-D evidence.
Do not modify the final exported code/parameters after the matching delegation;
the host checks that the identical exported code and parameters really ran in
Playground. A stopped or failed attempt remains recorded. Execution provenance
is not task success: report the native result honestly even if still unresolved.

The supplied helpers are public task knowledge, not earlier learned programs. Use only helpers applicable to the current native task. Use fresh public observations and current task instructions to establish object identity and progress. Occlusion means unknown. EE pose convergence or gripper closure alone is not task success. Only the unchanged native checker establishes whole-task success. Any internal completed/needs_verification status without native success triggers outside help.

Write scratch/task_results.json with native outcome, program path, all failed
attempts/revisions and evidence. Save reusable findings in lessons/. Then call
python scripts/env.py finish. Do not falsely claim a runnable exported program
when only isolated motion snippets were tested.

## Test: the fixed controller normally runs without System 2

The same sealed skills/task_program.json and its code start each fresh scene.
OpenRouter System 1 handles normal observation/parameter adaptation. You are invoked only after
failure, no progress, insufficient evidence or a declared decision/action/time
threshold. Read the specific scratch/handoff_*.json supplied in your startup.
The same physical/Python session continues; no reset or cross-scene learning.

During repair, skills/ and lessons/ are read-only. Write ONLY scene-local code and
one proposal under submission/, or an explicit unresolved-failure proposal.
Use zero-action exec to inspect evidence/geometry. Do NOT delegate, move or call
finish; the host controller dispatches the proposal to OpenRouter System 1 and closes the scene.
Do not lower confidence, enlarge the base limits, weaken the physical goal, or
replace a meaningful stage by repeated single-action restarts. Subsequent scenes
always receive the original frozen base, never this scene's repaired version.
