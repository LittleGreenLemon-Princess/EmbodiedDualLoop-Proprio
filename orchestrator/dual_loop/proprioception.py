"""Public measured state and action effects; never infer grasp success.

Inspired by ASPIRE's observation/feedback separation. No ASPIRE code is copied.
"""
from __future__ import annotations

import copy
import time

import numpy as np


def vector(value, size=None):
    try:
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or not array.size or not np.isfinite(array).all():
            return None
        if size is not None and array.size != size:
            return None
        return array.tolist()
    except (ValueError, TypeError):
        return None


class Proprioception:
    """Read only the existing public RoboDojo state, not simulator internals.

    Timestamp denotes host acquisition, not a sensor capture timestamp. Velocities
    and torques are deliberately absent until a documented public mapping exists.
    """

    def __init__(self, environment, clock=time.monotonic):
        self.environment, self.clock = environment, clock
        self.last_action = None
        self.action_samples = 0

    def read(self):
        state = self.environment.public_observation().get('state', {})
        arms = {}
        for arm in ('left', 'right'):
            pose = vector(state.get(arm + '_ee_pose'), 7)
            if pose is not None:
                norm = np.linalg.norm(pose[3:])
                pose = pose[:3] + (np.array(pose[3:]) / norm).tolist() if norm > 1e-12 else None
            jaws = vector(state.get(arm + '_ee_joint_state'))
            opening = jaws[0] if jaws else None
            if opening is not None and not 0 <= opening <= 1:
                opening = None
            joints = vector(state.get(arm + '_arm_joint_state'))
            values = {'ee_pose_world_wxyz': pose, 'joint_position_rad': joints,
                      'gripper_opening_normalized': opening}
            arms[arm] = {**values, 'valid': {k: v is not None for k, v in values.items()}}
        return {'schema': 'public-proprioception-v1', 'acquired_at_monotonic_s': self.clock(),
                'timestamp_meaning': 'host acquisition; sensor capture age is unknown',
                'source': 'public_observation.state',
                'units': {'translation': 'm', 'joint_position': 'rad',
                          'quaternion': 'wxyz', 'gripper': 'normalized: 0=closed, 1=open'},
                'arms': arms}

    def record_action(self, before, after):
        effects = {}
        for arm in ('left', 'right'):
            old, new = before['arms'][arm], after['arms'][arm]
            effect = {}
            for field in ('joint_position_rad', 'gripper_opening_normalized'):
                a, b = old[field], new[field]
                if a is None or b is None or np.shape(a) != np.shape(b):
                    effect[field + '_delta'] = None
                else:
                    delta = np.asarray(b) - np.asarray(a)
                    effect[field + '_delta'] = delta.tolist()
            a, b = old['ee_pose_world_wxyz'], new['ee_pose_world_wxyz']
            effect['translation_delta_m'] = (
                (np.array(b[:3]) - a[:3]).tolist() if a is not None and b is not None else None)
            effect['rotation_delta_rad'] = (
                float(2 * np.arccos(np.clip(abs(np.dot(a[3:], b[3:])), 0, 1)))
                if a is not None and b is not None else None)
            effects[arm] = effect
        self.action_samples += 1
        self.last_action = {'before': before, 'after': after, 'effects': effects,
                            'interpretation': 'Measured change only; no contact or grasp proof.'}

    def snapshot(self):
        return {**self.read(), 'action_samples': self.action_samples,
                'last_action': copy.deepcopy(self.last_action)}
