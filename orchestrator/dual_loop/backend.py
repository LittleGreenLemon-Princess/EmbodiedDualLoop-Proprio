"""Run parameterized Python inside the existing persistent four-primitive sandbox."""

from __future__ import annotations

from environments import execute


class ProgramBackend:
    def __init__(self, environment, brief, *, gripper_feedback=False, proprioception=False):
        if not hasattr(environment, 'bounded_actions') or not hasattr(environment, 'program_value'):
            raise ValueError('Dual loop requires the bounded CodeTask execution backend')
        self.env, self.brief = environment, brief
        from .gripper_state import GripperFeedback
        self.grippers = GripperFeedback(environment) if gripper_feedback else None
        from .proprioception import Proprioception
        self.proprioception = Proprioception(environment) if proprioception else None
        self.experiment_condition = ('dual-loop-proprio-v1' if proprioception else 'dual-loop-v1')

    def _execute(self, code, limit):
        with self.env.bounded_actions(limit):
            result = execute(self.env, code)
        return result, self.env._program_action_calls

    def initialize(self):
        code = '\n\n'.join(item['code'] for item in self.brief.sources)
        code += '\n\ndl_state = skill_init(' + ascii(self.brief.parameters) + ')\n'
        code += "dl_feedback = {}\n" + self._snapshot_code()
        result, calls = self._execute(code, 0)
        return self._snapshot(result), result, calls

    def refresh(self):
        result, calls = self._execute(self._snapshot_code(), 0)
        return self._snapshot(result), result, calls

    def tick(self, option):
        before = self.grippers.read() if self.grippers else None
        body_before = self.proprioception.read() if self.proprioception else None
        code = 'dl_feedback = skill_tick(dl_state, ' + ascii(option) + ')\n'
        result, calls = self._execute(code + self._snapshot_code(), 1)
        if self.proprioception and calls:
            self.proprioception.record_action(body_before, self.proprioception.read())
        if self.grippers and calls:
            self.grippers.record_action(before)
        return self._snapshot(result), result, calls

    @staticmethod
    def _snapshot_code():
        return ("dl_snapshot = {'context': skill_context(dl_state), "
                "'options': skill_options(dl_state), 'feedback': dl_feedback}\n")

    def _snapshot(self, result):
        if not result.ok:
            return None
        try:
            value = self.env.program_value('dl_snapshot')
            if (not isinstance(value, dict) or not isinstance(value.get('context'), dict)
                    or not isinstance(value.get('options'), dict)):
                raise ValueError('Skill must return context and options objects')
            if any(not isinstance(key, str) or not key for key in value['options']):
                raise ValueError('Action IDs must be nonempty strings')
            if self.grippers:
                # Measured public state, independent of generated skill context.
                measured = self.grippers.snapshot()
                value['context']['measured_grippers'] = measured
                if isinstance(value.get('feedback'), dict):
                    value['feedback']['measured_grippers'] = measured
            if self.proprioception:
                import copy
                value = copy.deepcopy(value)
                measured = self.proprioception.snapshot()
                value['context']['proprioception'] = measured
                if isinstance(value.get('feedback'), dict):
                    value['feedback']['proprioception'] = measured
            return value
        except (ValueError, TypeError, KeyError) as error:
            result.ok = False
            result.stderr += '\nInvalid skill snapshot: ' + str(error)
            return None
