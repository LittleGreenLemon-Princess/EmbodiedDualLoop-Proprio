"""Real restricted-code execution and runtime handoffs, without GPU or network."""

import json
import contextlib
import io
import runpy
import tempfile
import unittest
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from environments import execute
from orchestrator.dual_loop import Decision, ExecutorLoop, TaskBrief
from orchestrator.dual_loop.backend import ProgramBackend
from orchestrator.dual_loop.policies import AdvancePolicy, JevPolicy
from orchestrator.session import Session
from scripts.smoke_dual_loop import CpuTask, DemoPolicy, rim_sources


class Policy:
    def __init__(self, decision):
        self.decision = decision

    def select(self, state, options):
        return self.decision


class DualLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.brief = TaskBrief('rim-pick', 'Lift and verify the bowl.', rim_sources(), parameters={
            'detection': {'ok': True, 'center': [-.274, -.063, .815], 'radius': .059},
            'max_actions': 150,
        }, max_action_calls=150, max_decisions=160)

    def run_loop(self, *, env=None, policy=None, brief=None, **kwargs):
        env = env or CpuTask()
        brief = brief or self.brief
        loop = ExecutorLoop(brief, ProgramBackend(env, brief), policy or AdvancePolicy(),
                            self.root / 'loop', **kwargs)
        return loop.run(), env, loop

    def test_incremental_pick_stops_for_verification_and_preserves_unused_arm(self):
        report, env, loop = self.run_loop(policy=DemoPolicy())
        self.assertEqual(report['status'], 'awaiting_verification')
        self.assertFalse(report['subtask_verified'])
        self.assertFalse(report['latest']['context']['grasp_verified'])
        self.assertEqual(report['action_calls'], len(env.actions))
        self.assertEqual(report['latest']['context']['adjustments'], 2)
        self.assertGreater(len(env.actions), 20)
        for action in env.actions:
            np.testing.assert_allclose(action['right_ee_pose'], [.30, -.35, .92, 1, 0, 0, 0])
            self.assertAlmostEqual(action['right_ee_joint_state'][0], .3)
        stages = [e['effect']['event']['stage_before'] for e in loop.events
                  if e['kind'] == 'action']
        first_close = stages.index('close')
        self.assertTrue(all(a['left_ee_joint_state'][0] == 1 for a in env.actions[:first_close - 1]))
        self.assertTrue((self.root / 'loop/report.json').is_file())

    def test_every_tick_has_at_most_one_action_and_bounded_motion(self):
        report, env, loop = self.run_loop()
        for event in loop.events:
            if event['kind'] == 'action':
                self.assertLessEqual(event['action_calls'], 1)
                effect = event['effect']['event']
                self.assertLessEqual(np.linalg.norm(np.array(effect['command_pose'][:3])
                                                   - effect['before_pose'][:3]), .01000001)
        self.assertEqual(report['status'], 'awaiting_verification')

    def test_stuck_actuator_escalates_before_closing(self):
        report, env, _ = self.run_loop(env=CpuTask(stuck=True))
        self.assertEqual(report['status'], 'needs_system2')
        self.assertEqual(report['reason'], 'no_progress')
        self.assertLess(report['action_calls'], 20)
        self.assertTrue(all(a['left_ee_joint_state'][0] == 1 for a in env.actions))

    def test_native_budget_ends_loop_without_more_actions(self):
        report, env, _ = self.run_loop(env=CpuTask(limit=2))
        self.assertEqual(report['reason'], 'native_episode_ended')
        self.assertEqual(len(env.actions), 2)

    def test_low_confidence_or_unavailable_option_never_moves(self):
        for decision, reason in [(Decision('advance', .1), 'low_confidence'),
                                 (Decision('invented'), 'selector_error')]:
            with self.subTest(reason=reason):
                self.root = self.root / reason
                report, env, _ = self.run_loop(policy=Policy(decision))
                self.assertEqual(report['reason'], reason)
                self.assertEqual(len(env.actions), 0)

    def test_model_done_is_not_accepted_as_physical_success(self):
        report, env, _ = self.run_loop(policy=Policy(Decision('advance', judgement='done')))
        self.assertEqual(report['status'], 'awaiting_verification')
        self.assertFalse(report['subtask_verified'])
        self.assertEqual(len(env.actions), 0)

    def test_missing_selector_credentials_is_identifiable_without_leaking_secrets(self):
        with patch.dict('os.environ', {}, clear=True), patch('urllib.request.OpenerDirector.open') as http:
            report, env, loop = self.run_loop(policy=JevPolicy())
        error = next(e for e in loop.events if e['kind'] == 'selector_error')
        self.assertEqual(error['error_type'], 'SelectorConfigurationError')
        self.assertEqual(error['selector'], 'JevPolicy')
        self.assertEqual(error['failure_stage'], 'select')
        self.assertFalse(error['robot_action_applied'])
        self.assertNotIn('message', error)
        self.assertEqual(report['reason'], 'selector_error')
        self.assertEqual(len(env.actions), 0)
        http.assert_not_called()

    def test_invalid_decision_is_distinguished_from_provider_failure(self):
        _, env, loop = self.run_loop(policy=Policy(Decision('invented')))
        error = next(e for e in loop.events if e['kind'] == 'selector_error')
        self.assertEqual(error['failure_stage'], 'validate_decision')
        self.assertEqual(len(env.actions), 0)

    def test_slow_decision_is_discarded_before_action(self):
        now = [0.]

        class SlowPolicy:
            def select(self, state, options):
                now[0] += 2.
                return Decision('advance')

        report, env, _ = self.run_loop(policy=SlowPolicy(), clock=lambda: now[0])
        self.assertEqual(report['reason'], 'stale_observation')
        self.assertEqual(len(env.actions), 0)

    def test_host_action_and_decision_limits_are_independent_of_skill_limits(self):
        for brief, reason in [(replace(self.brief, max_action_calls=2), 'action_call_budget'),
                              (replace(self.brief, max_decisions=2), 'decision_budget')]:
            with self.subTest(reason=reason):
                self.root = self.root / reason
                report, env, _ = self.run_loop(brief=brief)
                self.assertEqual(report['reason'], reason)
                self.assertEqual(len(env.actions), 2)

    def test_initialization_and_observation_cannot_move(self):
        code = '''
def skill_init(p):
    try:
        step({})
    except Exception:
        pass
    return {}
def skill_context(s):
    return {'status': 'active'}
def skill_options(s):
    return {'advance': 'continue'}
def skill_tick(s, option):
    return {}
'''
        brief = replace(self.brief, sources=({'path': 'skill.py', 'code': code},))
        report, env, _ = self.run_loop(brief=brief)
        self.assertEqual(report['reason'], 'program_error')
        self.assertEqual(len(env.actions), 0)
        self.assertIn('limit', (self.root / 'loop/000000_initialize/stderr.txt').read_text())

    def test_saved_step_alias_cannot_bypass_one_action_limit_even_with_caught_error(self):
        env = CpuTask()
        execute(env, 'saved_step = step')
        code = '''
def skill_init(p):
    return {}
def skill_context(s):
    return {'status': 'active'}
def skill_options(s):
    return {'advance': 'continue'}
def skill_tick(s, option):
    for i in range(3):
        try:
            saved_step({})
        except Exception:
            pass
    return {}
'''
        brief = replace(self.brief, sources=({'path': 'skill.py', 'code': code},))
        report, env, _ = self.run_loop(env=env, brief=brief)
        self.assertEqual(report['reason'], 'program_error')
        self.assertEqual(len(env.actions), 1)

    def test_reset_is_rejected_inside_delegation_even_in_playground(self):
        env = CpuTask()
        env.execution_phase = 'playground'
        code = '''
def skill_init(p):
    reset()
    return {}
def skill_context(s):
    return {'status': 'active'}
def skill_options(s):
    return {'advance': 'continue'}
def skill_tick(s, option):
    return {}
'''
        brief = replace(self.brief, sources=({'path': 'skill.py', 'code': code},))
        report, env, _ = self.run_loop(env=env, brief=brief)
        self.assertEqual(report['reason'], 'program_error')
        self.assertEqual(len(env.actions), 0)

    def test_parameter_rotation_recomputes_finger_contact(self):
        env = CpuTask()
        backend = ProgramBackend(env, self.brief)
        backend.initialize()
        ns = env._exec_globals
        state = ns['dl_state']
        state['stage'] = 'align'
        state['params'].update(pitch_deg=76., yaw_deg=92., roll_deg=2.)
        target = ns['rim_loop_target'](state)
        w, x, y, z = target[3:]
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        contact = np.array(target[:3]) + rotation @ [.145, 0, -.0032]
        np.testing.assert_allclose(contact, [-.218, -.063, .809], atol=1e-12)

    def test_adjustment_threshold_requests_help(self):
        env = CpuTask()
        brief = replace(self.brief, parameters={**self.brief.parameters, 'max_adjustments': 1})
        backend = ProgramBackend(env, brief)
        snapshot, _, _ = backend.initialize()
        while snapshot['context']['stage'] != 'align':
            snapshot, _, _ = backend.tick('advance')
        snapshot, _, calls = backend.tick('x_plus')
        self.assertEqual(calls, 1)
        self.assertEqual(snapshot['context']['reason'], 'adjustment_budget')

    def test_initial_closed_gripper_and_lost_alignment_do_not_trigger_closure(self):
        env = CpuTask()
        env.state['left_ee_joint_state'][:] = 0
        env.observe(env._observation_value())
        report, env, _ = self.run_loop(env=env)
        self.assertEqual(report['reason'], 'open_gripper_precondition_missing')
        self.assertEqual(len(env.actions), 0)

    def test_lost_alignment_before_close_does_not_close(self):
        env = CpuTask()
        backend = ProgramBackend(env, self.brief)
        snapshot, _, _ = backend.initialize()
        while snapshot['context']['stage'] != 'close':
            snapshot, _, _ = backend.tick('advance')
        previous = len(env.actions)
        env.state['left_ee_pose'][0] += .02
        env.observe(env._observation_value())
        snapshot, _, calls = backend.tick('advance')
        self.assertEqual(calls, 0)
        self.assertEqual(len(env.actions), previous)
        self.assertEqual(snapshot['context']['reason'], 'alignment_lost_before_close')

    def test_workspace_clients_expose_the_same_delegate_command(self):
        root = Path(__file__).resolve().parents[1]
        canonical = (root / 'orchestrator/session_client.py').read_text()
        self.assertEqual((root / 'tasks/RoboDojo/Workspace/scripts/env.py').read_text(), canonical)
        baseline = runpy.run_path(str(root / 'tasks_baseline/RoboDojo/Workspace/scripts/env.py'))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            baseline['main'](['delegate', '--help'])
        self.assertEqual(result.exception.code, 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            baseline['main'](['reset'])
        self.assertEqual(result.exception.code, 2)
        with self.assertRaises(ValueError):
            baseline['source_snapshot'](self.root, 'skills/extra.py', [])

    def test_sources_are_validated_before_execution(self):
        with self.assertRaises(ValueError):
            replace(self.brief, sources=({'path': 'empty.py', 'code': 'pass'},))
        for limit in [0, True, float('nan')]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                replace(self.brief, max_decisions=limit)

    def test_session_handoff_uses_one_outer_request_and_keeps_feedback(self):
        episode = self.root / 'episode'
        (episode / 'workspace').mkdir(parents=True)
        session = Session(CpuTask(), episode, phase='test', seed=0, budget=2,
                          seconds=None, execution_seconds=10)
        with patch('orchestrator.dual_loop.evidence.write_spooled_videos', return_value={}):
            result = session.handle({'op': 'delegate', 'task': asdict(self.brief),
                                     'selector': {'kind': 'advance'}})
        self.assertEqual(result['loop_protocol'], 'dual-loop-v1')
        self.assertEqual(result['executions_used'], 1)
        self.assertGreater(result['native_steps_used'], 20)
        self.assertFalse(result['success'])
        self.assertEqual(result['system1']['status'], 'awaiting_verification')
        self.assertTrue((episode / 'workspace' / result['report']).is_file())
        self.assertTrue((episode / 'workspace/observations/000001/observation.json').is_file())


class JevTests(unittest.TestCase):
    def test_live_api_rounded_distribution_is_preserved(self):
        answer = {'type': 'choice', 'choice': 'a', 'confidence': .79,
                  'probabilities': {'a': .81, 'b': .11, 'c': .04, 'd': .03}}
        JevPolicy._validate_distribution(answer, {'a', 'b', 'c', 'd'})
        self.assertAlmostEqual(sum(answer['probabilities'].values()), .99)
        answer['probabilities']['a'] = .6
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            JevPolicy._validate_distribution(answer, {'a', 'b', 'c', 'd'})

    def test_proxy_is_scoped_to_jev_and_does_not_inherit_global_proxy(self):
        with patch.dict('os.environ', {'HTTPS_PROXY': 'http://unrelated:9999'}, clear=True):
            direct = JevPolicy()
            self.assertFalse(any(isinstance(h, urllib.request.ProxyHandler)
                                 for h in direct._opener.handlers))
        with patch.dict('os.environ', {'TYPESAFE_PROXY_URL': 'http://127.0.0.1:2080'}, clear=True):
            proxied = JevPolicy()
            handlers = [h for h in proxied._opener.handlers
                        if isinstance(h, urllib.request.ProxyHandler)]
            self.assertEqual(handlers[0].proxies, {'https': 'http://127.0.0.1:2080'})

    def response(self, payload):
        answers = {}
        for key, question in payload['questions'].items():
            choice = 'advance' if key == 'next_action' else 'continue'
            answers[key] = {'type': 'choice', 'choice': choice, 'confidence': 1.,
                            'probabilities': {name: float(name == choice)
                                              for name in question['criteria']}}
        return {'model': 'jev-1.13.0', 'answers': answers}

    def test_request_batches_action_and_current_status_and_contains_no_key(self):
        captured = []

        def transport(payload):
            captured.append(payload)
            return self.response(payload)

        policy = JevPolicy(transport=transport)
        decision = policy.select({'current': {'stage': 'align'}},
                                 {'advance': 'continue', 'escalate': 'help'})
        self.assertEqual(decision.action, 'advance')
        self.assertEqual(set(captured[0]['questions']), {'next_action', 'current_status'})
        self.assertNotIn('Authorization', json.dumps(captured))

    def test_invalid_distribution_is_rejected(self):
        def transport(payload):
            value = self.response(payload)
            value['answers']['next_action']['probabilities']['advance'] = float('nan')
            return value

        with self.assertRaises(ValueError):
            JevPolicy(transport=transport).select({}, {'advance': 'continue'})


if __name__ == '__main__':
    unittest.main()
