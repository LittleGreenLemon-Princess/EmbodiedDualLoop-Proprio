"""Supplied, action-free diagnostic for a possibly frozen visual fingertip.

Only compare measured, synchronized endpoints of the same rigid fingertip.
Frame origins/axes may differ, but both frames must be stationary and metric.
Missing prerequisites produce unknown. No result authorizes motion or success.
"""
import math


def _vmg_finite(value):
    if isinstance(value, (bool, str)):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _vmg_vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(_vmg_finite(x) for x in value))


def visual_motion_consistency(before, after, *, now_s, clock_id,
                              max_age_s, max_pair_skew_s, max_interval_s,
                              min_ee_motion_m, max_static_tip_motion_m,
                              position_error_bound_m, orientation_error_bound_rad,
                              tip_offset_bound_m,
                              max_rotation_rad, max_position_norm_m):
    """Check one completed-motion interval, using explicitly calibrated limits.

    Each sample has ee/tip_position_m, ee_orientation_wxyz, ee/tip_capture_time_s,
    ee/tip_arm, ee/tip_frame, tip_id, clock_id, units='m', and true assertions
    timestamps_verified, stationary_frames, rigid_tip. These assertions must
    come from observation provenance/calibration, not from this helper.
    position_error_bound_m bounds each position measurement, including temporal
    alignment/calibration error. orientation_error_bound_rad bounds each pose.
    tip_offset_bound_m bounds the rigid EE-to-tip distance (not equality).
    """
    limits = [now_s, max_age_s, max_pair_skew_s, max_interval_s, min_ee_motion_m,
              max_static_tip_motion_m, position_error_bound_m, orientation_error_bound_rad,
              tip_offset_bound_m,
              max_rotation_rad, max_position_norm_m]
    if (not all(_vmg_finite(x) for x in limits) or now_s < 0 or max_age_s <= 0
            or max_pair_skew_s < 0 or max_interval_s <= 0
            or not 0 <= max_static_tip_motion_m < min_ee_motion_m
            or position_error_bound_m < 0 or tip_offset_bound_m < 0
            or not 0 <= orientation_error_bound_rad <= math.pi
            or not 0 <= max_rotation_rad <= math.pi or max_position_norm_m <= 0
            or not isinstance(clock_id, str) or not clock_id.strip()):
        raise ValueError('Require finite, ordered metric/time limits and a clock identity')
    result = {'status': 'unknown', 'reason': 'missing_sample',
              'task_success_verified': False}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return result
    for sample in (before, after):
        for flag in ('timestamps_verified', 'stationary_frames', 'rigid_tip'):
            if sample.get(flag) is not True:
                return dict(result, reason=flag + '_unverified')
        for key in ('ee_arm', 'tip_arm', 'ee_frame', 'tip_frame', 'tip_id'):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                return dict(result, reason='missing_identity_or_frame')
        if sample.get('units') != 'm':
            return dict(result, reason='metric_units_unverified')
        if sample.get('clock_id') != clock_id:
            return dict(result, reason='clock_mismatch')
        if sample['ee_arm'] != sample['tip_arm']:
            return dict(result, reason='arm_mismatch')
        for key in ('ee_position_m', 'tip_position_m'):
            if (not _vmg_vector(sample.get(key), 3)
                    or math.hypot(*sample[key]) > max_position_norm_m):
                return dict(result, reason='invalid_position_or_scale')
        q = sample.get('ee_orientation_wxyz')
        if not _vmg_vector(q, 4) or abs(math.hypot(*q) - 1.0) > 0.001:
            return dict(result, reason='invalid_orientation')
        times = [sample.get(key) for key in ('ee_capture_time_s', 'tip_capture_time_s')]
        if not all(_vmg_finite(t) for t in times):
            return dict(result, reason='missing_capture_time')
        if any(t < 0 or not 0 <= now_s - t <= max_age_s for t in times):
            return dict(result, reason='stale_or_future_sample')
        if abs(times[0] - times[1]) > max_pair_skew_s:
            return dict(result, reason='unsynchronized_pair')
    for key in ('ee_arm', 'tip_arm', 'ee_frame', 'tip_frame', 'tip_id'):
        if before[key] != after[key]:
            return dict(result, reason='identity_or_frame_changed')
    for key in ('ee_capture_time_s', 'tip_capture_time_s'):
        if not 0 < after[key] - before[key] <= max_interval_s:
            return dict(result, reason='unordered_or_long_interval')
    # Require separated capture windows; overlapping before/after pairs are ambiguous.
    if max(before['ee_capture_time_s'], before['tip_capture_time_s']) >= min(
            after['ee_capture_time_s'], after['tip_capture_time_s']):
        return dict(result, reason='overlapping_capture_windows')
    qa, qb = before['ee_orientation_wxyz'], after['ee_orientation_wxyz']
    dot = abs(sum((a / math.hypot(*qa)) * (b / math.hypot(*qb)) for a, b in zip(qa, qb)))
    angle = 2 * math.acos(min(1.0, dot))  # q and -q describe the same orientation.
    angle_bound = min(math.pi, angle + 2 * orientation_error_bound_rad)
    if angle_bound > max_rotation_rad:
        return dict(result, reason='orientation_changed', rotation_delta_rad=angle)
    ee_motion = math.dist(before['ee_position_m'], after['ee_position_m'])
    tip_motion = math.dist(before['tip_position_m'], after['tip_position_m'])
    # A rotating EE can translate while its rigid fingertip remains nearly fixed.
    rotation_bound = 2 * tip_offset_bound_m * math.sin(angle_bound / 2)
    # Both displacement magnitudes have two endpoint errors: subtract four bounds.
    lower_bound = ee_motion - rotation_bound - 4 * position_error_bound_m
    if not all(_vmg_finite(x) for x in (ee_motion, tip_motion, rotation_bound, lower_bound)):
        return dict(result, reason='nonfinite_motion_bound')
    result.update(ee_displacement_m=ee_motion, visual_tip_displacement_m=tip_motion,
                  rotation_delta_rad=angle, rotation_upper_bound_rad=angle_bound,
                  rotation_motion_bound_m=rotation_bound,
                  required_tip_motion_lower_bound_m=lower_bound)
    if ee_motion < min_ee_motion_m or lower_bound <= max_static_tip_motion_m:
        return dict(result, reason='insufficient_translation_evidence')
    if tip_motion <= max_static_tip_motion_m:
        return dict(result, status='suspected_static_visual_tip', reason='ee_moved_tip_static')
    return dict(result, status='no_static_tip_evidence', reason='static_tip_test_not_triggered')
