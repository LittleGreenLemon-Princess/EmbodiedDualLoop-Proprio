"""Public gripper control-setting feedback; no independent jaw measurement."""
import numpy as np


class GripperFeedback:
    def __init__(self, environment):
        self.environment = environment
        self.initial = self.read()
        self.last = {}
        self.stationary = {'left': 0, 'right': 0}
        self.actions = 0

    def read(self):
        state = self.environment.public_observation().get('state', {})
        measured = {}
        for arm in ('left', 'right'):
            try:
                value = np.asarray(state[arm + '_ee_joint_state'], dtype=float).reshape(-1)
                measured[arm] = float(value[0]) if value.size and np.isfinite(value[0]) else None
            except (KeyError, ValueError, TypeError):
                measured[arm] = None
        return measured

    def record_action(self, before):
        after = self.read()
        self.actions += 1
        for arm in ('left', 'right'):
            delta = (None if before[arm] is None or after[arm] is None
                     else after[arm] - before[arm])
            self.stationary[arm] = (self.stationary[arm] + 1
                                    if delta is not None and abs(delta) <= 1e-3 else 0)
            self.last[arm] = {'before': before[arm], 'after': after[arm],
                              'delta': delta,
                              'motion': ('unknown' if delta is None else
                                         'stationary' if abs(delta) <= 1e-3 else
                                         'closing' if delta < 0 else 'opening')}

    def snapshot(self):
        current = self.read()
        return {'schema': 'measured-grippers-v1',
                'source': 'public_observation.state.*_ee_joint_state[0]',
                'convention': 'RoboDojo ARX: 1=open target, 0=closed target. Values are '
                              'normalized last controller targets, not independently '
                              'measured jaw openings or force; targets may be rate-limited.',
                'interpretation': 'Legacy measured_opening and initial_opening keys contain '
                                  'reported control settings. Delta, closing/opening/stationary '
                                  'labels and stationary_action_count describe changes to '
                                  'those settings, not jaw motion or convergence. Independent '
                                  'jaw opening is unavailable; these fields do not establish '
                                  'contact, a grasp, or task success. Use public visual evidence.',
                'action_samples': self.actions,
                'arms': {arm: {'measured_opening': current[arm],
                              'initial_opening': self.initial[arm],
                              'last_action': self.last.get(arm),
                              'stationary_action_count': self.stationary[arm]}
                         for arm in ('left', 'right')}}
