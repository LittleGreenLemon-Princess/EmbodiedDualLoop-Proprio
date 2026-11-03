import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from integrations.skills.visual_motion_guard import visual_motion_consistency
from orchestrator.dual_loop.staging import stage_workspace


class VisualMotionGuardTests(unittest.TestCase):
    def setUp(self):
        self.before = dict(ee_position_m=[0, 0, 0], tip_position_m=[1, 0, 0],
                           ee_orientation_wxyz=[1, 0, 0, 0], ee_arm='left', tip_arm='left',
                           ee_frame='world', tip_frame='stationary_camera', tip_id='left_pad',
                           clock_id='sensor_clock', units='m', timestamps_verified=True,
                           stationary_frames=True, rigid_tip=True,
                           ee_capture_time_s=10.0, tip_capture_time_s=10.0)
        self.after = dict(self.before, ee_position_m=[.04, 0, 0],
                          ee_capture_time_s=10.2, tip_capture_time_s=10.2)
        self.limits = dict(now_s=10.3, clock_id='sensor_clock', max_age_s=1,
                           max_pair_skew_s=.01, max_interval_s=.5,
                           min_ee_motion_m=.015, max_static_tip_motion_m=.002,
                           position_error_bound_m=.0005, orientation_error_bound_rad=.001,
                           tip_offset_bound_m=.1,
                           max_rotation_rad=.05, max_position_norm_m=5)

    def diagnose(self, before=None, after=None, **limits):
        return visual_motion_consistency(self.before if before is None else before,
                                         self.after if after is None else after,
                                         **dict(self.limits, **limits))

    def test_frozen_tip_and_moving_tip_never_authorize_action_or_success(self):
        frozen = self.diagnose()
        self.assertEqual(frozen['status'], 'suspected_static_visual_tip')
        moved = self.diagnose(after=dict(self.after, tip_position_m=[1, .04, 0]))
        self.assertEqual(moved['status'], 'no_static_tip_evidence')
        for result in (frozen, moved):
            self.assertFalse(result['task_success_verified'])
            self.assertNotIn('allow_visual_correction', result)
            self.assertNotIn('action', result)
            json.dumps(result, allow_nan=False)

    def test_missing_and_unverified_prerequisites_are_unknown(self):
        for key in self.before:
            sample = dict(self.before)
            del sample[key]
            with self.subTest(key=key):
                self.assertEqual(self.diagnose(before=sample)['status'], 'unknown')
        for flag in ('timestamps_verified', 'stationary_frames', 'rigid_tip'):
            self.assertEqual(self.diagnose(after=dict(self.after, **{flag: False}))['status'], 'unknown')

    def test_arm_track_frame_units_and_clock_changes_are_unknown(self):
        for key, value in [('tip_arm', 'right'), ('tip_id', 'other_pad'),
                           ('tip_frame', 'wrist_camera'), ('clock_id', 'read_clock'), ('units', 'mm')]:
            with self.subTest(key=key):
                self.assertEqual(self.diagnose(after=dict(self.after, **{key: value}))['status'], 'unknown')

    def test_timestamp_stale_future_skew_reuse_and_overlap(self):
        cases = [dict(ee_capture_time_s=8), dict(tip_capture_time_s=10.4),
                 dict(tip_capture_time_s=10.22),
                 dict(ee_capture_time_s=10, tip_capture_time_s=10)]
        for change in cases:
            self.assertEqual(self.diagnose(after=dict(self.after, **change))['status'], 'unknown')
        before = dict(self.before, tip_capture_time_s=10.008)
        after = dict(self.after, ee_capture_time_s=10.005, tip_capture_time_s=10.009)
        self.assertEqual(self.diagnose(before, after)['reason'], 'overlapping_capture_windows')
        self.assertEqual(self.diagnose(max_interval_s=.1)['reason'], 'unordered_or_long_interval')

    def test_nonfinite_shape_and_scale_are_unknown(self):
        for value in (None, [0, 0], [0, math.nan, 0], [0, math.inf, 0],
                      [0, True, 0], ['0', 0, 0], [1000, 0, 0], {0: 0, 1: 0, 2: 0}, {0, 1, 2}):
            with self.subTest(value=value):
                self.assertEqual(self.diagnose(after=dict(self.after, tip_position_m=value))['status'], 'unknown')
        for q in ([0, 0, 0, 0], [2, 0, 0, 0], [math.nan, 0, 0, 0]):
            self.assertEqual(self.diagnose(after=dict(self.after, ee_orientation_wxyz=q))['reason'], 'invalid_orientation')

    def test_rotation_cancellation_is_not_misdiagnosed_as_static_tracking(self):
        angle = .04
        after = dict(self.after, ee_position_m=[.02, 0, 0],
                     ee_orientation_wxyz=[math.cos(angle/2), 0, 0, math.sin(angle/2)])
        result = self.diagnose(after=after, tip_offset_bound_m=1)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'insufficient_translation_evidence')
        after['ee_orientation_wxyz'] = [math.cos(.2/2), 0, 0, math.sin(.2/2)]
        self.assertEqual(self.diagnose(after=after)['reason'], 'orientation_changed')

    def test_quaternion_sign_and_unknown_with_insufficient_signal(self):
        self.assertEqual(self.diagnose(after=dict(self.after, ee_orientation_wxyz=[-1, 0, 0, 0]))['status'],
                         'suspected_static_visual_tip')
        self.assertEqual(self.diagnose(after=dict(self.after, ee_position_m=[.001, 0, 0]))['status'], 'unknown')
        self.assertEqual(self.diagnose(position_error_bound_m=.02)['status'], 'unknown')
        self.assertEqual(self.diagnose(orientation_error_bound_rad=.1)['reason'], 'orientation_changed')

    def test_invalid_limits_raise_before_any_inference(self):
        for key, value in [('max_age_s', 0), ('now_s', math.nan), ('max_pair_skew_s', -1),
                           ('min_ee_motion_m', .001), ('max_static_tip_motion_m', -.1),
                           ('tip_offset_bound_m', -1), ('max_rotation_rad', math.inf),
                           ('orientation_error_bound_rad', -1),
                           ('max_position_norm_m', True), ('clock_id', '')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.diagnose(**{key: value})

    def test_inputs_are_not_mutated(self):
        original = copy.deepcopy((self.before, self.after, self.limits))
        self.diagnose()
        self.assertEqual(original, (self.before, self.after, self.limits))

    def test_staged_helper_executes_in_robot_sandbox_without_actions(self):
        from simulators.interactive import SAFE_BUILTINS, validate_snippet
        with tempfile.TemporaryDirectory() as folder:
            stage_workspace(Path(folder), frozen_task=True, native_task='cover_blocks')
            helper = Path(folder) / 'skills/visual_motion_guard.py'
            code = helper.read_text(encoding='utf-8')
            code += '\nresult = visual_motion_consistency(' + repr(self.before) + ', ' + repr(self.after)
            code += ', **' + repr(self.limits) + ')\n'
            validate_snippet(code)
            namespace = {'__builtins__': dict(SAFE_BUILTINS)}
            exec(compile(code, '<guard-cpu-test>', 'exec'), namespace)
            self.assertEqual(namespace['result']['status'], 'suspected_static_visual_tip')
            guard = namespace['visual_motion_consistency']
            for sample, reason in [
                (dict(self.after, ee_capture_time_s=None), 'missing_capture_time'),
                (dict(self.after, tip_position_m=[10**400, 0, 0]), 'invalid_position_or_scale'),
            ]:
                with self.subTest(reason=reason):
                    result = guard(self.before, sample, **self.limits)
                    self.assertEqual(result['status'], 'unknown')
                    self.assertEqual(result['reason'], reason)
                    self.assertFalse(result['task_success_verified'])
            for invalid in (None, 10**400):
                with self.subTest(invalid_limit=invalid), self.assertRaises(ValueError):
                    guard(self.before, self.after, **dict(self.limits, max_age_s=invalid))
            self.assertIn('visual_motion_guard.py', (Path(folder)/'skills/SUPPLIED.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
