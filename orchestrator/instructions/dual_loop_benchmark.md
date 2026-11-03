# System 2: solve the current native RoboDojo task with Jev System 1

Phase: {{PHASE}}. Experimental condition: dual-loop-v1, general native-task run.
Read primitives/primitives.md, then obtain get_instruction() and current public
observations with a zero-action exec. Solve THAT unchanged native task. There is
no mandatory bowl-grasp, rotation preflight, artificial perturbation or handoff
probe. Do not spend the native budget on unrelated diagnostics.

## Your job and Jev's job

Maintain scratch/plan.json: ordered meaningful subtasks, dependencies, evidence,
current stage and revision. Your loop is plan -> generate parameterized code ->
delegate -> review raw evidence -> advance or repair. You own perception code,
geometry, code generation/debugging, parameter bounds and workflow changes.
Jev owns repeated observation, bounded option selection, action execution and
progress judgement WITHIN a delegated subtask. This is a real running experiment:
write code, execute it and inspect actual feedback.

ALL movement must use `python scripts/env.py delegate ... --selector jev` or
`--selector jev_factorized`. Ordinary exec is host-limited to ZERO actions and is
for observations, instruction and geometry only. No direct motor loop, reset,
alternate robot connection, advance baseline selector or privileged simulator
access. Do not inspect parent folders, other scenes, credentials or prior runs.
Never coach the next signed direction/step through task prose, context or option
descriptions. Pose state and servo descriptions are canonicalized by the host.

Delegate a meaningful goal, not repeated one-primitive tasks. Repeated closing,
settling and holding belong inside ONE delegation. After low confidence, do not
restart the same operation by changing its name, counter, initial stage or budget.
Repair an evidenced sensing/code/geometry/workflow issue, or preserve the failure.
Do not narrow competing actions to the next answer or lower confidence thresholds
to force progress. Set min_confidence >= .5, timeout 8 seconds, measured-state age
10 seconds; use reasonable per-subtask budgets within the native allowance.

## Parameterized programs

Create your own code in submission/. Each program defines exactly these entries:

```python
def skill_init(parameters):
    # Initialize persistent skill state; ZERO actions.
    return {...}
def skill_context(state):
    # Fresh public measurements, goal, progress and uncertainty; ZERO actions.
    return {...}
def skill_options(state):
    # Explicit bounded parameter options, including escalate; ZERO actions.
    return {...}
def skill_tick(state, option):
    # Apply the selected option; at most ONE native step(action).
    return {...}
```

Submit JSON with task_id, revision, goal, entry, includes, parameters,
max_decisions, max_action_calls, max_seconds, max_state_age_seconds and
min_confidence. Includes and entry must be workspace paths. Read supplied code
before using it. Code receives np and the documented primitives, no imports,
filesystem or network. Keep the unused arm at its measured pose and preserve
the gripper's reported control setting, not an independently measured jaw opening.
Persist state across ticks; return context.status='needs_verification' when the
stage needs external physical review, or a non-active status with a reason when
it cannot continue. A model saying done does not certify manipulation success.

Use Bash timeout 600000 ms for synchronous delegations. A client timeout does not
undo physical actions: inspect status and current feedback before resubmitting.
Read observations/<execution>/system1/report.json, system1/events.jsonl and the
sibling system1-api.jsonl. Inspect the raw camera images and public RGB-D as needed.

## Supplied tools and their limits

skills/ee_pose_servo.py and pose_correction.py support free-space known-target
EE pose motion. Wrap pose_loop_start/context/options/tick as the four entries.
Parameters: target_pose (metres, quaternion wxyz), arm, mode='servo' or 'correct',
allow_translation, allow_rotation, max_adjustments. Use --selector jev for servo;
jev_factorized selects direction and variable size for free-space correction.
World rotations left-multiply delta quaternions. Required rotation error is
target * inverse(measured): apply its sign, do not negate it again. Never label
contact insertion as free-space hover to bypass a precondition.

The bowl helpers apply ONLY to shallow upright WHITE bowls with ARX X5 geometry.
They are not general-purpose grasp detectors. For suitable bowls, include in order:
ee_pose_servo.py, bowl_rim_grasp.py, grasp_feedback.py, rim_pick_loop.py and your
wrapper (stepwise_pick.py shows the entries). Detect the bowl from current RGB-D;
pass feedback_enabled=True to rim_loop_start. Public feedback distinguishes a
stationary visible bowl from one following the rising EE, and stops an observed
empty grasp. Occlusion or unreliable fitting means UNKNOWN, never success or a
definite empty grasp. Inspect object_feedback and actual frames. For other objects,
write suitable public-observation feedback rather than reusing white-bowl logic.

Approaching an EE target or commanding gripper close does not establish a grasp.
Check actual object lift and subsequent following/holding while physics advances.
Record action-count/time span of hold; waiting for an API response is not a hold.
For placement/stacking, inspect the task-specific final object relation and the
unchanged native success signal. Keep native success separate from subtask claims.

## Phase rules and completion

During playground you may save reusable code in skills/ and findings in lessons/.
During test these directories are FROZEN/READ-ONLY; put scene-specific code in
submission/ and analysis in scratch/. Do not reset Test or transfer scene-specific
programs/state to other scenes. No cross-task knowledge transfer is implied.

Preserve every failed attempt and revision. Do not retry until success, change
native limits, loosen goals or hide model errors. If an operation remains
unresolved, report its actual failure/unknown outcome and budget. Starting slots:
{{EXECUTION_BUDGET}}. Native action allowance: {{NATIVE_ACTION_LIMIT}}.
Write scratch/task_results.json with native instruction, plan/subtask outcomes,
all revisions and failure reasons, object evidence references, native task success,
budgets and any unattempted goals. Call `python scripts/env.py finish` when done.
