import contextlib
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestrator import session_client, session_runner
from orchestrator.config import CODE_ROOT, load_task
from orchestrator.dual_loop.policies import AdvancePolicy
from orchestrator.frozen_program import load_program, validate_development_export
from orchestrator.session import Session
from scripts.smoke_dual_loop import CpuTask


class OpenRouterRoutingTests(unittest.TestCase):
    def test_actual_delegation_context_is_required_before_openrouter_seal(self):
        selector = {'kind': 'openrouter', 'model': 'deepseek/deepseek-v4.1-flash',
                    'effort': 'high', 'timeout': 120}
        code = ('def skill_init(p): return {}\n'
                'def skill_context(s): return {"status": "active"}\n'
                'def skill_options(s): return {"advance": "one bounded step"}\n'
                'def skill_tick(s, option):\n'
                '    obs = get_observation()\n'
                '    step({k: np.array(v).copy() for k,v in obs["state"].items()})\n'
                '    return {}\n')
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            workspace = episode/'workspace'
            for name in ('skills', 'lessons', 'submission', 'scratch'):
                (workspace/name).mkdir(parents=True)
            (workspace/'skills/task.py').write_text(code, encoding='utf-8')
            manifest_path = workspace/'skills/task_program.json'
            manifest = {'schema': 'frozen-task-v1', 'native_task': 'cover_blocks',
                        'selector': selector, 'observation_profile': 'proprioception-v1',
                        'task': {'task_id': 'whole-task', 'entry': 'skills/task.py',
                                 'goal': 'CPU bounded diagnostic', 'max_decisions': 1,
                                 'max_action_calls': 1, 'min_confidence': .5}}
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            program = load_program(workspace, 'cover_blocks')
            env = CpuTask()
            session = Session(env, episode, phase='playground', seed=0, budget=10,
                              seconds=None, execution_seconds=30, require_delegate=True,
                              selector_condition={k: selector[k] for k in ('kind', 'model', 'effort')})
            request = {'task': asdict(program['brief']), 'selector': selector,
                       'observation_profile': 'proprioception-v1'}
            for changes in ({'kind': 'jev'}, {'model': 'another-model'}, {'effort': 'low'}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'must match'):
                    session.delegate({**request, 'selector': {**selector, **changes}})
            self.assertEqual(env.control_steps, 0)
            with patch('orchestrator.dual_loop.policies.make_policy', return_value=AdvancePolicy()), \
                    patch('orchestrator.dual_loop.evidence.write_spooled_videos', return_value={}):
                session.delegate(request)
            self.assertEqual(env.control_steps, 1)
            seal = validate_development_export(workspace, 'cover_blocks')
            self.assertIn('development_context_sha256', seal)
            for changes in ({'observation_profile': 'baseline'},
                            {'selector': {**selector, 'timeout': 119}}):
                manifest_path.write_text(json.dumps({**manifest, **changes}), encoding='utf-8')
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_development_export(workspace, 'cover_blocks')
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            report_path = workspace/'observations/000001/system1/report.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            for changes in ({'reason': 'selector_error'}, {'action_calls': 0}):
                report_path.write_text(json.dumps({**report, **changes}), encoding='utf-8')
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_development_export(workspace, 'cover_blocks')

    def test_unported_external_worker_is_rejected_before_workspace_or_native_start(self):
        cfg = load_task('configs/dual-loop-openrouter-cover-blocks')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root/'profile.json'
            profile.write_text(json.dumps({'sif': '/unstarted/image.sif'}), encoding='utf-8')
            with patch.dict(os.environ, {'EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE': str(profile)}):
                with self.assertRaisesRegex(ValueError, 'direct Linux runner'):
                    session_runner.run_episode(cfg, root/'scene', root, phase='playground', seed=0, seconds=10)
            self.assertFalse((root/'scene').exists())

    def test_direct_linux_episode_waits_for_socket_before_offline_container(self):
        cfg = load_task('configs/dual-loop-openrouter-cover-blocks')
        started = []
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            session_runner.prepare_inputs(cfg, experiment)
            run = experiment/'playground'

            def spawn(command, **_kwargs):
                module = command[command.index('-m')+1]
                started.append(module)
                if module == 'orchestrator.session':
                    # Reproduce the session's readiness contract without physics.
                    Path(command[command.index('--socket')+1]).touch()
                    os.write(int(command[command.index('--ready-fd')+1]), b'{"event":"ready"}\n')
                    (run/'session-result.json').write_text(json.dumps({
                        'valid': True, 'closed': True, 'reason': 'finished'}), encoding='utf-8')
                    return SimpleNamespace(pid=9101, returncode=None, poll=lambda: None)
                self.assertEqual(module, 'orchestrator.openrouter_agent_host')
                return SimpleNamespace(pid=9102, returncode=0, poll=lambda: 0)

            with patch('orchestrator.launch_agent.require_image'), \
                    patch.object(session_runner.subprocess, 'Popen', side_effect=spawn), \
                    patch.object(session_runner.subprocess, 'run', return_value=SimpleNamespace(
                        returncode=0, stdout='', stderr='')) as docker, \
                    patch.object(session_runner, 'stop_process'), \
                    patch.object(session_runner, 'gpu_lease', return_value=contextlib.nullcontext()):
                result = session_runner.run_episode(cfg, run, experiment, phase='playground', seed=0, seconds=10)
            self.assertEqual(started, ['orchestrator.session', 'orchestrator.openrouter_agent_host'])
            self.assertIsNone(result['failure'])
            self.assertEqual(docker.call_args_list[0].args[0][:2], ['docker', 'run'])

    def test_opt_in_prompt_preserves_default_supplied_helpers(self):
        cfg = load_task('configs/dual-loop-openrouter-cover-blocks')
        session_runner.validate(cfg)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            session_runner.prepare_inputs(cfg, base/'openrouter')
            original = replace(cfg, agent=replace(cfg.agent, preset='claude', isolation='local'))
            session_runner.prepare_inputs(original, base/'original')
            for phase in ('playground', 'test'):
                new = (base/'openrouter/inputs'/f'{phase}.md').read_text(encoding='utf-8')
                old = (base/'original/inputs'/f'{phase}.md').read_text(encoding='utf-8')
                self.assertIn('openrouter-only-dual-loop-adaptation-v1', new)
                self.assertNotIn('openrouter-only-dual-loop-adaptation-v1', old)
            helper_bytes = lambda root: {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
            self.assertEqual(helper_bytes(base/'openrouter/inputs/skills'),
                             helper_bytes(base/'original/inputs/skills'))
        for changes in ({'isolation': 'local'}, {'network': 'bridge'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                session_runner.validate(replace(cfg, agent=replace(cfg.agent, **changes)))

    def test_solve_uses_host_api_but_frozen_test_starts_host_controller(self):
        cfg = load_task('configs/dual-loop-openrouter-cover-blocks')
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'frozen-task-seal.json').write_text(json.dumps({'program_sha256': 'sealed'}), encoding='utf-8')
            with patch('orchestrator.openrouter_agent_host.start_shell_container',
                       return_value=('offline-shell', '/tmp/session.sock')) as start:
                container, command = session_runner.agent_command(cfg, run, run/'socket', 'playground')
                self.assertEqual(container, 'offline-shell')
                self.assertIn('orchestrator.openrouter_agent_host', command)
                self.assertEqual(command[command.index('--model')+1], cfg.agent.model)
                self.assertEqual(start.call_args.kwargs['phase'], 'playground')
                start.reset_mock()
                container, command = session_runner.agent_command(cfg, run, run/'socket', 'test')
                self.assertIsNone(container)
                self.assertIn('orchestrator.frozen_task', command)
                start.assert_not_called()

    def test_cli_provider_defaults_do_not_send_jev_model_to_openrouter(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace/'submission').mkdir()
            (workspace/'submission/task.py').write_text('# source snapshot\n', encoding='utf-8')
            (workspace/'submission/task.json').write_text(json.dumps({
                'entry': 'submission/task.py', 'goal': 'current native goal'}), encoding='utf-8')
            with patch.dict(os.environ, {'EMBODIEDEVO_WORKSPACE': str(workspace)}), \
                    patch.object(session_client, 'request', return_value={'ok': True}) as request, \
                    contextlib.redirect_stdout(io.StringIO()):
                for provider, expected in [
                    ('jev', {'kind': 'jev', 'model': 'jev-1.13.0', 'timeout': 8.}),
                    ('openrouter', {'kind': 'openrouter', 'model': 'deepseek/deepseek-v4.1-flash',
                                    'timeout': 120., 'effort': 'high'})]:
                    with self.subTest(provider=provider):
                        self.assertEqual(session_client.main([
                            'delegate', 'submission/task.json', '--selector', provider]), 0)
                        self.assertEqual(request.call_args.args[0]['selector'], expected)
                        self.assertEqual(request.call_args.args[0]['task']['sources'][0]['code'],
                                         '# source snapshot\n')
        self.assertEqual((CODE_ROOT/'orchestrator/session_client.py').read_bytes(),
                         (CODE_ROOT/'tasks/RoboDojo/Workspace/scripts/env.py').read_bytes())

    def test_frozen_manifest_keeps_openrouter_and_observation_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace/'skills').mkdir()
            (workspace/'skills/task.py').write_text(
                'def skill_init(p): return {}\n'
                'def skill_context(s): return {}\n'
                'def skill_options(s): return {"advance": "bounded action"}\n'
                'def skill_tick(s, option): return {}\n', encoding='utf-8')
            selector = {'kind': 'openrouter', 'model': 'deepseek/deepseek-v4.1-flash',
                        'effort': 'high', 'timeout': 120}
            (workspace/'skills/task_program.json').write_text(json.dumps({
                'schema': 'frozen-task-v1', 'native_task': 'cover_blocks',
                'selector': selector, 'observation_profile': 'proprioception-v1',
                'task': {'task_id': 'whole-task', 'entry': 'skills/task.py',
                         'goal': 'current native goal', 'min_confidence': .5}}), encoding='utf-8')
            program = load_program(workspace, 'cover_blocks')
            self.assertEqual(program['selector'], selector)
            self.assertEqual(program['observation_profile'], 'proprioception-v1')
            cfg = load_task('configs/dual-loop-openrouter-cover-blocks')
            session_runner.validate_frozen_selector(cfg, program)
            for changes in ({'kind': 'jev'}, {'model': 'another-model'}, {'effort': 'low'}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'must match'):
                    session_runner.validate_frozen_selector(cfg, {**program, 'selector': {**selector, **changes}})


if __name__ == '__main__':
    unittest.main()
