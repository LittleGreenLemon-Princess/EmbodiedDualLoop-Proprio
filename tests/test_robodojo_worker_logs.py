"""Persist real child-process diagnostics without Docker, models or a simulator."""

import contextlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import os
from unittest.mock import Mock, patch

from integrations.robodojo_policy import worker
from orchestrator.config import config_dict, load_task
from orchestrator.session_runner import prepare_inputs


class WorkerLogTests(unittest.TestCase):
    def exercise(self, *, flat=False, exit_code=1, startup_error=None, cleanup_error=None,
                 container='test-agent'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cfg = load_task('tasks_baseline/RoboDojo')
        prepare_inputs(cfg, root)
        run = root if flat else root / 'episodes/episode_0000'
        command = [sys.executable, '-c',
                   "import sys; print('agent started'); "
                   "sys.stderr.write('x' * 262144 + '\\nprovider connection failed\\n'); "
                   f"sys.exit({exit_code})"]
        connection = Mock()
        session = SimpleNamespace(closed=False, valid=True, seconds=None, reason=None, save=Mock())
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        with contextlib.ExitStack() as patches:
            for target, options in [
                ('agent_command', {'side_effect': startup_error} if startup_error else
                                  {'return_value': (container, command)}),
                ('BridgeTask', {'return_value': SimpleNamespace(success=False, control_steps=0)}),
                ('Session', {'return_value': session}),
                ('threading.Thread', {}),
                ('codex_token_usage', {'return_value': {}}),
            ]:
                patches.enter_context(patch(f'integrations.robodojo_policy.worker.{target}', **options))
            cleanup = patches.enter_context(patch.object(worker.subprocess, 'run', side_effect=cleanup_error))
            worker.run_worker(connection, {'run_dir': str(root), 'task_config': config_dict(cfg),
                              'phase': 'test', 'flat_episode': flat}, {}, 100, 0)
            if container is None or startup_error:
                cleanup.assert_not_called()
            else:
                cleanup.assert_called_once()
                self.assertEqual(cleanup.call_args.args[0], ['docker', 'rm', '-f', container])
        runtime = json.loads((run / ('agent/runtime.json' if flat else 'runtime.json')).read_text())
        return run, runtime, connection, session

    def test_nonzero_exit_preserves_full_stderr_and_context_in_both_layouts(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                run, runtime, connection, session = self.exercise(flat=flat)
                self.assertEqual(runtime['status'], 'failed')
                self.assertEqual(runtime['agent_exit_code'], 1)
                self.assertEqual(runtime['phase'], 'test')
                self.assertEqual(runtime['env_task'], 'stack_bowls')
                self.assertIn('Agent exited with status 1', runtime['error'])
                self.assertGreaterEqual(runtime['finished_at'], runtime['started_at'])
                self.assertEqual((run / runtime['stdout_log']).read_text(), 'agent started\n')
                stderr = run / runtime['stderr_log']
                self.assertEqual(stderr.read_text(), 'x' * 262144 + '\nprovider connection failed\n')
                self.assertEqual(stderr.stat().st_mode & 0o777, 0o600)
                self.assertFalse(stderr.is_relative_to(run / 'workspace'))
                self.assertFalse(session.valid)
                self.assertEqual(connection.send.call_args.args[0]['kind'], 'error')

    def test_startup_exception_is_saved_even_without_a_child_exit_code(self):
        run, runtime, _, _ = self.exercise(startup_error=PermissionError('Docker access denied'))
        self.assertEqual(runtime['status'], 'failed')
        self.assertIsNone(runtime['agent_exit_code'])
        self.assertIn('PermissionError: Docker access denied', runtime['error'])
        self.assertIn('model', runtime)
        self.assertEqual((run / runtime['stderr_log']).read_bytes(), b'')

    def test_stderr_alone_does_not_turn_a_clean_exit_into_failure(self):
        run, runtime, connection, session = self.exercise(exit_code=0)
        self.assertEqual(runtime['status'], 'finished')
        self.assertEqual(runtime['agent_exit_code'], 0)
        self.assertTrue((run / runtime['stderr_log']).is_file())
        self.assertTrue(session.valid)
        self.assertEqual(connection.send.call_args.args[0]['kind'], 'finished')

    def test_cleanup_timeout_preserves_completed_episode_and_records_container(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                run, runtime, connection, session = self.exercise(
                    flat=flat, exit_code=0,
                    cleanup_error=subprocess.TimeoutExpired(['docker', 'rm'], 30),
                )
                self.assertEqual(runtime['status'], 'finished')
                self.assertTrue(session.valid)
                self.assertEqual(connection.send.call_args.args[0]['kind'], 'finished')
                evidence = json.loads((run / 'agent/cleanup-error.json').read_text())
                self.assertEqual(evidence['container'], 'test-agent')
                self.assertIn('TimeoutExpired', evidence['error'])

    def test_host_agent_without_container_finishes_without_docker_cleanup(self):
        # Fixed-task and local agent_command branches both return (None, argv).
        # Exercise the real worker's exit path with that contract and a real child.
        for flat in (False, True):
            with self.subTest(flat=flat):
                run, runtime, connection, session = self.exercise(
                    flat=flat, exit_code=0, container=None)
                self.assertEqual(runtime['status'], 'finished')
                self.assertTrue(session.valid)
                self.assertIsNone(runtime['container'])
                self.assertEqual(connection.send.call_args.args[0]['kind'], 'finished')
                self.assertFalse((run / 'agent/cleanup-error.json').exists())


if __name__ == '__main__':
    unittest.main()
