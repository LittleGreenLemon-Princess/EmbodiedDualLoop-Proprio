"""Stage explicitly supplied helpers and System 2 instructions for the pilot."""
from pathlib import Path
import shutil


def stage_workspace(inputs, *, benchmark=False, frozen_task=False, native_task='',
                    selector_provider='jev'):
    root = Path(__file__).resolve().parents[2]
    name = ('dual_loop_openrouter_task.md' if frozen_task and selector_provider == 'openrouter'
            else 'dual_loop_frozen_task.md' if frozen_task else
            'dual_loop_benchmark.md' if benchmark else 'dual_loop.md')
    prompt = (root / 'orchestrator/instructions' / name).read_text(encoding='utf-8')
    prompt = prompt.replace('{{NATIVE_TASK_ID}}', native_task)
    for phase in ('playground', 'test'):
        (inputs / (phase + '.md')).write_text(prompt.replace('{{PHASE}}', phase), encoding='utf-8')
    skills = inputs / 'skills'
    skills.mkdir(exist_ok=True)
    for path in ['integrations/ee_pose_servo.py', 'integrations/bowl_rim_grasp.py',
                 'integrations/rim_pick_loop.py', 'integrations/skills/grasp_feedback.py', 'integrations/skills/thin_rim_grasp/stepwise_pick.py',
                 'integrations/skills/pose_correction.py',
                 'integrations/skills/visual_motion_guard.py']:
        shutil.copyfile(root / path, skills / Path(path).name)
    (skills / 'SUPPLIED.md').write_text(
        'These helpers are supplied task knowledge, not learned skills. '
        'System 2 must create its own task wrappers, parameters and plan. '
        'No earlier experiment programs or results are supplied.\n'
        'visual_motion_guard.py is an action-free supplied diagnostic, not a learned '
        'skill or a motion/success verifier. Include it explicitly before your own '
        'task program. It requires synchronized capture times, stationary metric '
        'frames, same-arm rigid-tip identity and calibrated uncertainty/rotation '
        'bounds; missing evidence returns unknown. Request System 2 review for '
        'unknown or suspected_static_visual_tip before further visual correction.\n')
