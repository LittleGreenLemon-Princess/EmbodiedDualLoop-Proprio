import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
from pathlib import Path

import numpy as np

from orchestrator.dual_loop import ExecutorLoop, TaskBrief
from orchestrator.dual_loop.backend import ProgramBackend
from orchestrator.dual_loop.policies import AdvancePolicy
from orchestrator.dual_loop.proprioception import Proprioception
from orchestrator.dual_loop.skill_library import SkillLibrary
from scripts.smoke_dual_loop import CpuTask, rim_sources


class ProprioceptionTests(unittest.TestCase):
    def test_missing_invalid_and_sign_equivalent_orientation(self):
        env = CpuTask()
        body = Proprioception(env)
        before = body.read()
        self.assertIsNone(before['arms']['left']['joint_position_rad'])
        env.state['left_ee_pose'][3:] *= -1
        env.observe(env._observation_value())
        body.record_action(before, body.read())
        self.assertAlmostEqual(body.last_action['effects']['left']['rotation_delta_rad'], 0)
        env.state['left_ee_pose'][0] = np.nan
        env.observe(env._observation_value())
        self.assertIsNone(body.read()['arms']['left']['ee_pose_world_wxyz'])
        json.dumps(body.snapshot(), allow_nan=False)

    def test_measured_action_effects_reach_selector_and_report_even_with_transform(self):
        env = CpuTask()
        brief = TaskBrief('test', 'Lift', rim_sources(), parameters={
            'detection': {'ok': True, 'center': [-.274, -.063, .815], 'radius': .059}},
            max_action_calls=2)
        seen = []

        class Policy(AdvancePolicy):
            def select(self, state, options):
                seen.append(copy.deepcopy(state))
                return super().select(state, options)

        with tempfile.TemporaryDirectory() as folder:
            loop = ExecutorLoop(brief, ProgramBackend(env, brief, proprioception=True),
                                Policy(), Path(folder) / 'run',
                                state_transform=lambda state: {'current': {'trusted': True}})
            report = loop.run()
            self.assertEqual(report['experiment_condition'], 'dual-loop-proprio-v1')
            self.assertIn('proprioception', seen[0])
            self.assertEqual(seen[1]['proprioception']['action_samples'], 1)
            evidence = report['latest']['context']['proprioception']
            self.assertEqual(evidence['action_samples'], 2)
            self.assertEqual(evidence['last_action']['effects']['right']['translation_delta_m'], [0, 0, 0])
            self.assertFalse(report['subtask_verified'])
            records = [json.loads(s) for s in (Path(folder) / 'run/events.jsonl').read_text().splitlines()]
            actions = [e for e in records if e['kind'] == 'action']
            self.assertIn('proprioception', actions[0]['effect'])

    def test_baseline_has_no_new_observation(self):
        env = CpuTask()
        brief = TaskBrief('test', 'Observe', rim_sources(), parameters={})
        backend = ProgramBackend(env, brief)
        self.assertIsNone(backend.proprioception)
        self.assertEqual(backend.experiment_condition, 'dual-loop-v1')


class SkillLibraryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.library = SkillLibrary(self.root)
        self.skill = self.library.skills[0]

    def compile(self, **overrides):
        args = dict(embodiment='robodojo-arx-x5', available_signals=self.skill['required_signals'],
                    parameters={}, task_id='library-test', goal='Observe then grasp')
        args.update(overrides)
        return self.library.compile('thin-rim-pick', '1.0.0', **args)

    def test_search_compile_and_version_lock(self):
        self.assertEqual(len(self.library.search(embodiment='robodojo-arx-x5', capabilities=['rim-grasp'])), 1)
        self.assertEqual(len(self.compile().sources), 4)
        hover = self.library.compile('ee-offset-correction', '1.0.0',
                                     embodiment='robodojo-arx-x5',
                                     available_signals=self.skill['required_signals'],
                                     parameters={'target_pose': [0, 0, 1, 1, 0, 0, 0]},
                                     task_id='hover', goal='Correct EE position')
        self.assertEqual(len(hover.sources), 2)
        with self.assertRaises(ValueError):
            self.library.compile('thin-rim-pick', '2.0.0', embodiment='robodojo-arx-x5',
                                 available_signals=[], parameters={}, task_id='test', goal='test')

    def test_scope_signals_tampering_and_path_escape(self):
        for args in ({'embodiment': 'yam'}, {'available_signals': []}):
            with self.assertRaises(ValueError):
                self.compile(**args)
        self.skill['sources'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.compile()
        with self.assertRaises(ValueError):
            self.library._path('../outside.py')

    def test_compiled_task_can_be_submitted_by_workspace_client(self):
        from orchestrator import session_client
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'submission/task.json'
            path.parent.mkdir()
            path.write_text(json.dumps(asdict(self.compile())), encoding='utf-8')
            with patch.dict('os.environ', {'EMBODIEDEVO_WORKSPACE': folder}), \
                 patch.object(session_client, 'request', return_value={'ok': True}) as request, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(session_client.main(['delegate', 'submission/task.json',
                    '--observation-profile', 'proprioception-v1']), 0)
            payload = request.call_args.args[0]
            self.assertEqual(payload['observation_profile'], 'proprioception-v1')
            self.assertEqual(len(payload['task']['sources']), 4)
