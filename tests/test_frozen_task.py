from dataclasses import asdict, replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from orchestrator import local_agent, session_runner
from orchestrator.config import config_dict, config_from_dict, load_task
from orchestrator.dual_loop.policies import AdvancePolicy
from orchestrator.dual_loop import Decision
from orchestrator.frozen_program import load_program, validate_development_export
from orchestrator.frozen_task import drive, knowledge_hashes, load_repair, RepairValidationError
from orchestrator.session import Session
from scripts.smoke_dual_loop import CpuTask


PROGRAM = '''
def skill_init(p):
    obs = get_observation()
    return {'target': float(obs['state']['left_ee_pose'][0]) + p['distance'],
            'step': p['step'], 'status': 'active'}
def skill_context(s):
    return {'status': s['status']}
def skill_options(s):
    return {'advance': 'One bounded waypoint', 'escalate': 'Stop'}
def skill_tick(s, option):
    obs = get_observation()
    action = {k: np.array(v).copy() for k, v in obs['state'].items()}
    action['left_ee_pose'][0] = min(s['target'], action['left_ee_pose'][0] + s['step'])
    step(action)
    return {'target': s['target']}
'''


class GoalCpu(CpuTask):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.resets = getattr(self, 'resets', 0) + 1
        self.state['left_ee_pose'][0] += (seed or 0) * .03
        self.target = float(self.state['left_ee_pose'][0]) + .02
        self.observe(self._observation_value())
        return self.public_observation(), {}

    def control_step(self, action):
        super().control_step(action)
        if self.state['left_ee_pose'][0] >= self.target - 1e-8:
            self.success = self.terminated = True
        return self.transition()


class FrozenTaskTests(unittest.TestCase):
    def test_only_bytecode_caches_are_excluded_from_fixed_knowledge_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'helper.py'; source.write_text('x=1')
            baseline = knowledge_hashes(root)
            cache = root/'__pycache__'; cache.mkdir()
            (cache/'helper.cpython-312.pyc').write_bytes(b'generated')
            self.assertEqual(knowledge_hashes(root), baseline)
            (cache/'hidden.py').write_text('x=2')
            self.assertNotEqual(knowledge_hashes(root), baseline)
            (cache/'hidden.py').unlink()
            source.write_text('x=3')
            self.assertNotEqual(knowledge_hashes(root), baseline)
            (cache/'linked.pyc').symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                knowledge_hashes(root)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = [patch('orchestrator.dual_loop.policies.make_policy',
                              return_value=AdvancePolicy()),
                        patch('orchestrator.dual_loop.evidence.write_spooled_videos', return_value={})]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def setup_scene(self, *, step=.01, seed=0, interventions=2):
        episode = self.root / f'episode_{len(list(self.root.iterdir()))}'
        w = episode / 'workspace'
        for part in ['skills', 'lessons', 'submission', 'scratch']:
            (w/part).mkdir(parents=True, exist_ok=True)
        (w/'skills/task_program.py').write_text(PROGRAM)
        manifest = {'schema': 'frozen-task-v1', 'native_task': 'general_pickup',
                    'max_system2_interventions': interventions,
                    'selector': {'kind': 'jev', 'timeout': 8},
                    'task': {'task_id': 'full-task', 'goal': 'Reach CPU goal',
                             'entry': 'skills/task_program.py', 'includes': [],
                             'parameters': {'distance': .02, 'step': step},
                             'max_decisions': 2, 'max_action_calls': 2,
                             'max_seconds': 60, 'max_state_age_seconds': 10,
                             'min_confidence': .5}}
        (w/'skills/task_program.json').write_text(json.dumps(manifest))
        env = GoalCpu()
        s = Session(env, episode, phase='test', seed=seed, budget=10, seconds=None,
                    execution_seconds=120, require_delegate=True, controller_key='test-controller')
        return episode, w, manifest, env, s

    def run_controller(self, episode, workspace, session, repair):
        program = load_program(workspace, 'general_pickup')
        return drive(workspace, 'general_pickup', program['sha256'], session.handle,
                     repair, episode/'fixed-task', controller_key='test-controller')

    def test_normal_path_uses_no_system2_and_same_program_reinitializes_each_scene(self):
        fingerprints, starts = [], []
        def forbidden(*args):
            self.fail('Normal path must not start System 2')
        for seed in [0, 3]:
            ep, w, _, env, session = self.setup_scene(seed=seed)
            fingerprints.append(load_program(w, 'general_pickup')['sha256'])
            starts.append(float(env.state['left_ee_pose'][0]))
            result = self.run_controller(ep, w, session, forbidden)
            self.assertTrue(result['native_success'])
            self.assertTrue(result['primary_success'])
            self.assertEqual(result['system2_interventions'], 0)
            self.assertEqual(env.control_steps, 2)
            self.assertTrue(session.closed)
            task = json.loads((w/'observations/000001/system1/task.json').read_text())
            self.assertEqual(task['goal'], env.task_prompt())
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(starts[0], starts[1])

    def test_failed_primary_repairs_same_physical_scene_without_promoting_code(self):
        ep, w, manifest, env, session = self.setup_scene(step=0.)
        resets = env.resets
        baseline = (w/'skills/task_program.json').read_bytes()
        def repair(path, handoff):
            self.assertEqual(handoff['trigger'], 'decision_budget')
            self.assertEqual(env.control_steps, 2)
            self.assertEqual(env.resets, resets)
            cache = w/'skills/__pycache__'; cache.mkdir()
            (cache/'helper.cpython-312.pyc').write_bytes(b'import-generated cache')
            (w/'submission/repair.py').write_text(PROGRAM)
            spec = dict(manifest['task'])
            spec.update(entry='submission/repair.py', parameters={'distance': .02, 'step': .01})
            p = w / handoff['proposal_path']
            p.write_text(json.dumps({'task': spec, 'selector': {'kind': 'jev', 'timeout': 8}}))
            return p
        result = self.run_controller(ep, w, session, repair)
        self.assertFalse(result['primary_success'])
        self.assertTrue(result['native_success'])
        self.assertEqual(result['system2_interventions'], 1)
        self.assertGreater(result['repair_agent_seconds'], 0)
        self.assertEqual(env.resets, resets)
        self.assertEqual(env.control_steps, 4)
        self.assertEqual((w/'skills/task_program.json').read_bytes(), baseline)
        self.assertTrue((w/'observations/000001/system1/report.json').exists())
        self.assertTrue((w/'observations/000002/system1/report.json').exists())
        # A new scene starts with the unchanged failing base, not this repair.
        _, w2, _, _, _ = self.setup_scene(step=0., seed=9)
        self.assertEqual(load_program(w2, 'general_pickup')['sha256'],
                         load_program(w, 'general_pickup')['sha256'])
        self.assertEqual(load_program(w2, 'general_pickup')['brief'].parameters['step'], 0.)

    def test_real_frozen_edit_is_rejected_but_repair_time_is_saved(self):
        ep, w, _, env, session = self.setup_scene(step=0.)
        def repair(path, handoff):
            (w/'skills/task_program.py').write_text(PROGRAM + '\n# forbidden edit\n')
            return None
        with self.assertRaisesRegex(ValueError, 'Frozen base knowledge changed'):
            self.run_controller(ep, w, session, repair)
        result = json.loads((ep/'fixed-task/result.json').read_text())
        self.assertGreater(result['repair_agent_seconds'], 0)
        self.assertGreater(result['elapsed_seconds'], 0)
        events = [json.loads(x) for x in (ep/'fixed-task/events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['kind'], 'system2_return')

    def test_repair_schema_and_utf8_sources_use_real_task_validation(self):
        _, w, manifest, _, _ = self.setup_scene()
        base = load_program(w, 'general_pickup')['brief']
        proposal = w/'submission/repair.json'
        source = w/'submission/repair.py'
        source.write_text(PROGRAM + '\n# 当前场景的局部修复\n', encoding='utf-8')
        invalid = [([], 'proposal'), ({'diagnosis': 'missing task'}, 'task'),
                   ({'task': []}, 'task'), ({'task': {}}, 'task.entry'),
                   ({'task': {'entry': 1}}, 'task.entry'),
                   ({'task': {'entry': 'submission/repair.py', 'includes': 'skills/a.py'}}, 'task.includes'),
                   ({'task': {'entry': 'submission/repair.py', 'includes': ['submission/repair.py']}}, 'task.includes'),
                   ({'task': {'entry': 'submission/repair.py', 'max_seconds': 10**400}}, 'task'),
                   ({'task': {'entry': '../secret.py'}}, 'task.sources'),
                   ({'task': {'entry': 'submission/missing.py'}}, 'task.sources')]
        for raw, field in invalid:
            with self.subTest(raw=raw):
                proposal.write_text(json.dumps(raw), encoding='utf-8')
                with self.assertRaises(RepairValidationError) as raised:
                    load_repair(w, proposal, base, 2, base_selector=manifest['selector'])
                self.assertEqual(raised.exception.field, field)
                self.assertNotIn(str(w), str(raised.exception))
        proposal.write_text(json.dumps({'task': {'entry': 'submission/repair.py',
                            'parameters': {'distance': .02, 'step': .01, '说明': '保持状态'}}},
                                      ensure_ascii=False), encoding='utf-8')
        repaired, _ = load_repair(w, proposal, base, 2, base_selector=manifest['selector'])
        self.assertEqual(repaired.parameters['说明'], '保持状态')
        self.assertIn('当前场景', repaired.sources[-1]['code'])
        source.write_text('def broken(:\n', encoding='utf-8')
        with self.assertRaises(RepairValidationError) as raised:
            load_repair(w, proposal, base, 2, base_selector=manifest['selector'])
        self.assertEqual(raised.exception.field, 'task')

    def test_omitted_repair_fields_inherit_current_revision_without_default_budget_increase(self):
        _, w, manifest, _, _ = self.setup_scene()
        base = load_program(w, 'general_pickup')['brief']
        helper = w/'skills/helper.py'; helper.write_text('GAIN = 1\n', encoding='utf-8')
        current = replace(base, parameters={'distance': .03, 'step': .005}, max_decisions=1,
                          max_action_calls=1, max_seconds=20, max_state_age_seconds=2,
                          min_confidence=.8, sources=({'path': 'skills/helper.py', 'code': 'GAIN = 1\n'}, *base.sources))
        (w/'submission/repair.py').write_text(PROGRAM, encoding='utf-8')
        proposal = w/'submission/repair.json'
        proposal.write_text(json.dumps({'task': {'entry': 'submission/repair.py'}}), encoding='utf-8')
        repaired, selector = load_repair(w, proposal, base, 2, current=current,
                                        base_selector=manifest['selector'])
        for field in ['parameters', 'max_decisions', 'max_action_calls', 'max_seconds',
                      'max_state_age_seconds', 'min_confidence']:
            self.assertEqual(getattr(repaired, field), getattr(current, field))
        self.assertEqual(repaired.sources[0]['path'].replace('\\', '/'), 'skills/helper.py')
        self.assertEqual(selector, manifest['selector'])
        self.assertEqual(repaired.revision, 2)

    def test_repair_cannot_change_selector_or_enlarge_frozen_limits(self):
        _, w, manifest, _, _ = self.setup_scene()
        base = load_program(w, 'general_pickup')['brief']
        (w/'submission/repair.py').write_text(PROGRAM, encoding='utf-8')
        proposal = w/'submission/repair.json'
        pinned = {'kind': 'jev_factorized', 'model': 'pinned-model', 'timeout': 4.}
        for changes in [{'kind': 'jev'}, {'model': 'other-model'}, {'timeout': 8.},
                        {'observation_profile': 'different'}]:
            with self.subTest(changes=changes):
                proposal.write_text(json.dumps({'task': {'entry': 'submission/repair.py'},
                                               'selector': {**pinned, **changes}}), encoding='utf-8')
                with self.assertRaisesRegex(RepairValidationError, 'frozen selector'):
                    load_repair(w, proposal, base, 2, base_selector=pinned)
        for field, value in [('max_decisions', 3), ('max_action_calls', 3),
                             ('max_seconds', 61), ('max_state_age_seconds', 11), ('min_confidence', .4)]:
            with self.subTest(field=field):
                proposal.write_text(json.dumps({'task': {'entry': 'submission/repair.py', field: value}}), encoding='utf-8')
                with self.assertRaises(RepairValidationError) as raised:
                    load_repair(w, proposal, base, 2, base_selector=pinned)
                self.assertEqual(raised.exception.field, 'task.' + field)
        proposal.write_text(json.dumps({'task': {'entry': 'submission/repair.py'}}), encoding='utf-8')
        self.assertEqual(load_repair(w, proposal, base, 2, base_selector=pinned)[1], pinned)

    def test_handoff_template_is_executable_and_invalid_schema_is_logged_without_replay(self):
        ep, w, _, env, session = self.setup_scene(step=0.)
        def repair(path, handoff):
            template = handoff['proposal_template']
            self.assertEqual(template['task']['max_decisions'], 2)
            self.assertEqual(template['task']['parameters']['step'], 0.)
            self.assertEqual(template['task']['entry'], 'skills/task_program.py')
            proposal = w/handoff['proposal_path']
            proposal.write_text(json.dumps(template), encoding='utf-8')
            base = load_program(w, 'general_pickup')
            parsed, selector = load_repair(w, proposal, base['brief'], 2, base_selector=base['selector'])
            self.assertEqual(parsed.max_action_calls, 2)
            self.assertEqual(selector, base['selector'])
            proposal.write_text('{"diagnosis":"sensitive-proposal-text"}', encoding='utf-8')
            return proposal
        result = self.run_controller(ep, w, session, repair)
        self.assertEqual(result['reason'], 'invalid_repair_proposal')
        self.assertEqual(result['delegations'], 1)
        self.assertEqual(env.control_steps, 2)
        events_text = (ep/'fixed-task/events.jsonl').read_text(encoding='utf-8')
        error = next(e for e in map(json.loads, events_text.splitlines()) if e['kind'] == 'invalid_repair')
        self.assertEqual(error['field'], 'task')
        self.assertIn('requires a task object', error['message'])
        self.assertNotIn('sensitive-proposal-text', events_text)

    def test_system2_cannot_dispatch_finish_or_bypass_zero_action_exec(self):
        ep, w, _, env, s = self.setup_scene()
        for request in [{'op': 'finish'}, {'op': 'delegate'},
                        {'op': 'finish', 'controller_key': 'wrong'}]:
            with self.assertRaises(PermissionError): s.handle(request)
        with patch('orchestrator.session.write_videos', return_value={}):
            response = s.handle({'op': 'exec', 'files': [{'path': 'submission/bad.py',
                                  'code': "step(get_observation()['state'])"}]})
        self.assertFalse(response['ok']); self.assertEqual(env.control_steps, 0)
        s.record_rpc({'op': 'status', 'controller_key': 'test-controller'}, s.status())
        self.assertNotIn('test-controller', (ep/'events.jsonl').read_text())

    def test_intervention_limit_and_unchanged_or_invalid_repair_preserve_failure(self):
        for mode in ['limit', 'unchanged', 'invalid']:
            ep, w, manifest, env, s = self.setup_scene(step=0., interventions=0 if mode=='limit' else 2)
            calls = []
            def repair(path, handoff):
                calls.append(handoff)
                proposal = w/handoff['proposal_path']
                proposal.write_text('not-json' if mode=='invalid' else json.dumps(
                    {'task': manifest['task'], 'selector': manifest['selector']}))
                return proposal
            result = self.run_controller(ep, w, s, repair)
            self.assertFalse(result['native_success'])
            self.assertEqual(result['reason'], {'limit':'system2_intervention_limit',
                'unchanged':'unchanged_repair_rejected', 'invalid':'invalid_repair_proposal'}[mode])
            self.assertEqual(env.control_steps, 2)
            self.assertEqual(len(calls), 0 if mode=='limit' else 1)

    def test_untested_or_modified_export_cannot_be_sealed(self):
        ep, w, _, env, s = self.setup_scene()
        s.controller_key = None
        s.phase = 'playground'
        with self.assertRaisesRegex(ValueError, 'not executed'):
            validate_development_export(w, 'general_pickup')
        program = load_program(w, 'general_pickup')
        s.development_export_task = 'general_pickup'
        with self.assertRaisesRegex(ValueError, 'not executed'):
            s.handle({'op': 'finish', 'controller_key': 'test-controller'})
        self.assertFalse(s.closed)
        s.handle({'op': 'delegate', 'task': asdict(program['brief']),
                  'selector': program['selector'], 'controller_key': 'test-controller'})
        seal = validate_development_export(w, 'general_pickup')
        self.assertEqual(seal['program_sha256'], program['sha256'])
        s.handle({'op': 'finish', 'controller_key': 'test-controller'})
        self.assertTrue(s.closed)
        (w/'skills/task_program.py').write_text(PROGRAM+'\n# Untested edit\n')
        with self.assertRaisesRegex(ValueError, 'not executed'):
            validate_development_export(w, 'general_pickup')
        with self.assertRaises(ValueError): load_program(w, 'stack_bowls')

    def test_test_driver_does_not_launch_development_agent(self):
        cfg = load_task('configs/dual-loop-frozen')
        self.assertEqual(config_from_dict(config_dict(cfg)), cfg)
        with tempfile.TemporaryDirectory() as t:
            run = Path(t); (run/'frozen-task-seal.json').write_text('{"program_sha256":"pinned"}')
            _, command = session_runner.agent_command(cfg, run, run/'socket', 'test')
            self.assertIn('orchestrator.frozen_task', command)
            self.assertNotIn('orchestrator.local_agent', command)
            _, command = session_runner.agent_command(cfg, run, run/'socket', 'playground')
            self.assertIn('orchestrator.local_agent', command)
            session_runner.prepare_inputs(cfg, run)
            prompt = (run/'inputs/playground.md').read_text()
            self.assertIn('Native task ID: general_pickup', prompt)
            self.assertIn('skills/task_program.json', prompt)
        prompt = local_agent.command('claude', 'model', 'test', role='repair',
                                     handoff='scratch/handoff_0001.json')[-1]
        self.assertIn('MUST NOT move, call delegate, reset or finish', prompt)

    def test_model_done_without_native_success_requires_handoff(self):
        ep, w, _, env, s = self.setup_scene()
        class Done:
            def select(self, state, options):
                return Decision('advance', judgement='done')
        with patch('orchestrator.dual_loop.policies.make_policy', return_value=Done()):
            result = self.run_controller(ep, w, s, lambda *a: None)
        self.assertFalse(result['native_success'])
        self.assertEqual(result['system2_interventions'], 1)
        self.assertEqual(result['primary_exit_reason'], 'physical_evidence_required')
        self.assertEqual(env.control_steps, 0)

    def test_summary_separates_primary_success_from_recovery_and_accounts_for_cost(self):
        cfg = load_task('configs/dual-loop-frozen')
        rows = {}
        for i, (success, primary, calls) in enumerate([(True, True, 0), (True, False, 1),
                                                     (False, False, 2)]):
            rows[i] = {'valid': True, 'task_completed': success,
                       'fixed_task': {'primary_success': primary, 'system2_interventions': calls,
                                      'repair_agent_seconds': calls*10.,
                                      'initial_program_sha256': 'same-base'}}
        report = session_runner.test_report(self.root, 3, rows)
        summary = report['fixed_task_summary']
        self.assertEqual(report['successes'], 2)
        self.assertEqual(summary['primary_successes'], 1)
        self.assertEqual(summary['recovered_successes'], 1)
        self.assertEqual(summary['system2_interventions'], 3)
        self.assertEqual(summary['repair_agent_seconds'], 30.)
        self.assertEqual(summary['initial_program_sha256'], ['same-base'])


if __name__ == '__main__':
    unittest.main()
