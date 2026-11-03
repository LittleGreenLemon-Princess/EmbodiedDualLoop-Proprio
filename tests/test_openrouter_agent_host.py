from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestrator import openrouter_agent_host as host
from orchestrator.openrouter_agent_host import parallel_tool_rejection_messages


class ParallelToolRejectionTests(unittest.TestCase):
    def test_rejects_every_parallel_call_without_running_any(self):
        calls = [{"id": "call-a"}, {"id": "call-b"}]

        messages = parallel_tool_rejection_messages(calls)

        self.assertEqual([message["tool_call_id"] for message in messages],
                         ["call-a", "call-b"])
        self.assertTrue(all(message["role"] == "tool" for message in messages))
        payloads = [json.loads(message["content"]) for message in messages]
        self.assertTrue(all("No tool in the parallel response was executed" in payload["error"]
                            for payload in payloads))
        self.assertTrue(all("do not replay actions" in payload["error"]
                            for payload in payloads))

    def test_refuses_to_build_rejection_for_a_single_call(self):
        with self.assertRaises(ValueError):
            parallel_tool_rejection_messages([{"id": "call-a"}])


class SessionClosedTests(unittest.TestCase):
    @contextmanager
    def agent(self, role='solve'):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / 'workspace').mkdir()
            (episode / 'workspace/instruction.md').write_text('CPU fixture', encoding='utf-8')
            args = SimpleNamespace(episode=episode, log_dir=episode / 'agent', role=role,
                phase='playground' if role == 'solve' else 'test', container='fake',
                container_socket='fake', image='fake', model='fixture', effort='high',
                max_rounds=2, handoff='submission/handoff.json')
            call = {'id': 'call-1', 'function': {'name': 'shell', 'arguments': '{"command":"inspect"}'}}
            replies = [
                {'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
                 'choices': [{'finish_reason': 'tool_calls',
                              'message': {'role': 'assistant', 'tool_calls': [call]}}]},
                {'choices': [{'finish_reason': 'stop',
                              'message': {'role': 'assistant', 'content': 'done'}}]},
            ]
            with patch.object(host, 'read_key', return_value='synthetic-test-key'), \
                 patch.object(host, 'openrouter_opener', return_value=object()), \
                 patch.object(host, 'session_deadline', return_value=None), \
                 patch.object(host.signal, 'signal'), \
                 patch.object(host, 'api_request', side_effect=replies) as api, \
                 patch.object(host, 'execute_tool', return_value=({'exit_code': 0}, [])) as tool, \
                 redirect_stdout(io.StringIO()):
                yield args, api, tool

    def test_closed_after_tool_stops_solve_and_repair_and_retains_usage(self):
        for role in ('solve', 'repair'):
            for valid in (True, False):
                with self.subTest(role=role, valid=valid), self.agent(role) as (args, api, tool):
                    state = {'closed': True, 'valid': valid, 'success': False,
                             'reason': 'agent_finished' if valid else 'execution_failed',
                             'private_scoring': 'must-stay-host-only'}
                    terminal = args.episode / 'session-result.json'
                    text = json.dumps(state)
                    def close_session(*_args):
                        terminal.write_text(text, encoding='utf-8')
                        return {'exit_code': 0}, []
                    tool.side_effect = close_session
                    self.assertEqual(host.run(args), 0)
                    self.assertEqual(api.call_count, 1)
                    tool.assert_called_once()
                    self.assertEqual(terminal.read_text(encoding='utf-8'), text)
                    usage = json.loads((args.log_dir / 'usage.json').read_text(encoding='utf-8'))
                    self.assertEqual((usage['receipts'], usage['total_tokens']), (1, 5))
                    transcript = (args.log_dir / 'openrouter-session.jsonl').read_text(encoding='utf-8')
                    event = json.loads(transcript.splitlines()[-1])
                    self.assertEqual(event['type'], 'session_closed')
                    self.assertEqual(event['payload']['valid'], valid)
                    self.assertNotIn('private_scoring', transcript)
                    self.assertNotIn('must-stay-host-only', transcript)

    def test_already_closed_makes_no_api_or_tool_call(self):
        for role in ('solve', 'repair'):
            with self.subTest(role=role), self.agent(role) as (args, api, tool):
                (args.episode / 'session-result.json').write_text('{"closed":true}', encoding='utf-8')
                self.assertEqual(host.run(args), 0)
                api.assert_not_called()
                tool.assert_not_called()
                usage = json.loads((args.log_dir / 'usage.json').read_text(encoding='utf-8'))
                self.assertEqual(usage['receipts'], 0)

    def test_missing_or_open_session_continues_even_after_native_termination(self):
        for role in ('solve', 'repair'):
            for state in (None, {'closed': False}, {'closed': False, 'terminated': True,
                                                   'truncated': True, 'success': True}):
                with self.subTest(role=role, state=state), self.agent(role) as (args, api, tool):
                    if state is not None:
                        (args.episode / 'session-result.json').write_text(json.dumps(state), encoding='utf-8')
                    # Neither writable workspace files nor arbitrary shell output are authoritative.
                    (args.episode / 'workspace/session-result.json').write_text('{"closed":true}', encoding='utf-8')
                    tool.return_value = ({'exit_code': 0, 'output': '{"closed":true}'}, [])
                    self.assertEqual(host.run(args), 0)
                    self.assertEqual(api.call_count, 2)
                    tool.assert_called_once()

    def test_malformed_authoritative_state_fails_before_api(self):
        for text in ('{', 'null', '[]', '{}', '{"closed":1}', '{"closed":"true"}'):
            with self.subTest(text=text), self.agent() as (args, api, tool):
                (args.episode / 'session-result.json').write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    host.run(args)
                api.assert_not_called()
                tool.assert_not_called()
                usage = json.loads((args.log_dir / 'usage.json').read_text(encoding='utf-8'))
                self.assertEqual(usage['receipts'], 0)

    def test_unreadable_authoritative_state_fails_before_api(self):
        with self.agent() as (args, api, tool):
            original = Path.read_text
            def read(path, *positional, **keywords):
                if path == args.episode / 'session-result.json':
                    raise PermissionError('fixture denied')
                return original(path, *positional, **keywords)
            with patch.object(Path, 'read_text', read), self.assertRaises(PermissionError):
                host.run(args)
            api.assert_not_called()
            tool.assert_not_called()
            self.assertTrue((args.log_dir / 'usage.json').is_file())


if __name__ == "__main__":
    unittest.main()
