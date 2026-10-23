"""CPU control-flow demonstration. No robot physics, object verification or model calls."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from orchestrator.dual_loop import Decision, ExecutorLoop, TaskBrief
from orchestrator.dual_loop.backend import ProgramBackend
from simulators.execution import CodeTask
from simulators.robodojo_api import RoboDojoLowLevelApi


def rim_sources():
    root = Path(__file__).resolve().parents[1]
    paths = ['integrations/ee_pose_servo.py', 'integrations/bowl_rim_grasp.py',
             'integrations/rim_pick_loop.py',
             'integrations/skills/thin_rim_grasp/stepwise_pick.py']
    return tuple({'path': path, 'code': (root / path).read_text()} for path in paths)


class CpuTask(CodeTask):
    """Instant waypoint follower for protocol tests, deliberately no object model."""

    def __init__(self, *, stuck=False, limit=200):
        super().__init__()
        self.max_steps = self.native_step_limit = limit
        self.stuck = stuck
        self.actions = []
        self.initialize(RoboDojoLowLevelApi(self))
        self.reset(seed=0)

    def reset(self, *, seed=None, options=None):
        self.actions = []
        self._task_prompt = 'CPU protocol demonstration only'
        self.state = {
            'left_ee_pose': np.array([-.30, -.35, .92, 1., 0., 0., 0.]),
            'right_ee_pose': np.array([.30, -.35, .92, 1., 0., 0., 0.]),
            'left_ee_joint_state': np.array([1.]), 'right_ee_joint_state': np.array([.3]),
        }
        return self.start_episode(seed, self._observation_value())

    def _observation_value(self):
        return {'state': copy.deepcopy(self.state), 'instruction': self._task_prompt,
                'vision': {'cam_head': {'color': np.zeros((8, 8, 3), dtype=np.uint8)}}}

    def control_step(self, action):
        if self.done:
            return self.transition()
        self.actions.append(copy.deepcopy(action))
        if not self.stuck:
            for key, value in action.items():
                self.state[key] = np.array(value, dtype=float)
        self.control_steps += 1
        self.truncated = self.control_steps >= self.max_steps
        self.observe(self._observation_value())
        return self.transition()


class DemoPolicy:
    """Scripted adjustments exercise the menu; this is not a perception policy."""

    def __init__(self):
        self.pending = ['finer', 'x_plus', 'pitch_minus']

    def select(self, state, options):
        if state['current']['stage'] == 'align' and self.pending:
            action = self.pending.pop(0)
            if action in options:
                return Decision(action, model='scripted-cpu-demo-v1')
        return Decision('advance', model='scripted-cpu-demo-v1')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stuck', action='store_true')
    parser.add_argument('--proprioception', action='store_true')
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.run_dir.name[:12], '%Y%m%d%H%M')
    except ValueError:
        parser.error('--run-dir must start with YYYYMMDDHHMM')
    env = CpuTask(stuck=args.stuck)
    brief = TaskBrief('cpu-rim-pick', 'Demonstrate incremental EE control; verify object lift separately.',
                      rim_sources(), parameters={
                          'detection': {'ok': True, 'center': [-.274, -.063, .815], 'radius': .059},
                          'max_actions': 150,
                      }, max_action_calls=150, max_decisions=160)
    loop = ExecutorLoop(brief, ProgramBackend(env, brief, proprioception=args.proprioception),
                        DemoPolicy(), args.run_dir)
    report = loop.run()
    print(json.dumps({
        'status': report['status'], 'reason': report['reason'],
        'experiment_condition': report['experiment_condition'],
        'decisions': report['decisions'], 'action_calls': report['action_calls'],
        'model_called': False, 'robot_physics': False, 'grasp_verified': False,
        'report': str((args.run_dir / 'report.json').resolve()),
    }, indent=2))
    expected = 'needs_system2' if args.stuck else 'awaiting_verification'
    return 0 if report['status'] == expected else 1


if __name__ == '__main__':
    raise SystemExit(main())
