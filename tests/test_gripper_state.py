"""Public gripper feedback survives stationary EE motion and observation-only refresh."""
import unittest
import numpy as np
from orchestrator.dual_loop.backend import ProgramBackend
from orchestrator.dual_loop.gripper_state import GripperFeedback
from orchestrator.dual_loop import TaskBrief
from scripts.smoke_dual_loop import CpuTask

PROGRAM = '''
def skill_init(p): return {}
def skill_context(s): return {'status': 'active'}
def skill_options(s): return {'advance': 'Close jaws at the measured EE pose'}
def skill_tick(s, option):
    obs = get_observation()
    a = {k: np.array(v).copy() for k, v in obs['state'].items()}
    a['left_ee_joint_state'][0] = max(0., a['left_ee_joint_state'][0] - .2)
    step(a)
    return {}
'''

class GripperStateTests(unittest.TestCase):
    def test_backend_injects_control_changes_and_keeps_refresh_action_free(self):
        env = CpuTask()
        brief = TaskBrief('close', 'Close and verify separately',
                          ({'path': 'close.py', 'code': PROGRAM},))
        backend = ProgramBackend(env, brief, gripper_feedback=True)
        start, result, calls = backend.initialize()
        self.assertTrue(result.ok); self.assertEqual(calls, 0)
        self.assertIsNone(start['context']['measured_grippers']['arms']['left']['last_action'])
        pose = env.state['left_ee_pose'].copy()
        for _ in range(2): snapshot, result, calls = backend.tick('advance')
        self.assertTrue(result.ok); self.assertEqual(calls, 1)
        measured = snapshot['context']['measured_grippers']
        self.assertEqual(snapshot['feedback']['measured_grippers'], measured)
        self.assertEqual(measured['action_samples'], 2)
        self.assertAlmostEqual(measured['arms']['left']['measured_opening'], .6)
        self.assertAlmostEqual(measured['arms']['left']['last_action']['delta'], -.2)
        self.assertEqual(measured['arms']['left']['last_action']['motion'], 'closing')
        self.assertEqual(measured['arms']['right']['stationary_action_count'], 2)
        self.assertIn('normalized last controller targets', measured['convention'])
        self.assertIn('not independently measured jaw openings', measured['convention'])
        self.assertIn('not jaw motion or convergence', measured['interpretation'])
        self.assertEqual(set(measured), {'schema', 'source', 'convention', 'interpretation',
                                        'action_samples', 'arms'})
        np.testing.assert_array_equal(env.state['left_ee_pose'], pose)
        refreshed, _, calls = backend.refresh()
        self.assertEqual(calls, 0)
        self.assertEqual(refreshed['context']['measured_grippers'], measured)
        self.assertEqual(env.control_steps, 2)
        self.assertNotIn('grasp_verified', measured)

    def test_missing_nan_and_stationary_are_distinct_from_closed(self):
        env = CpuTask(); tracker = GripperFeedback(env)
        before = tracker.read(); tracker.record_action(before)
        self.assertEqual(tracker.snapshot()['arms']['left']['last_action']['motion'], 'stationary')
        env.state['left_ee_joint_state'] = np.array([np.nan])
        env.state.pop('right_ee_joint_state'); env.observe(env._observation_value())
        tracker.record_action(before)
        for arm in tracker.snapshot()['arms'].values():
            self.assertIsNone(arm['measured_opening'])
            self.assertIsNone(arm['last_action']['delta'])
            self.assertEqual(arm['last_action']['motion'], 'unknown')
            self.assertEqual(arm['stationary_action_count'], 0)

    def test_original_profile_keeps_existing_context(self):
        env = CpuTask()
        brief = TaskBrief('close', 'Close', ({'path':'close.py', 'code':PROGRAM},))
        snapshot, _, _ = ProgramBackend(env, brief).initialize()
        self.assertNotIn('measured_grippers', snapshot['context'])
