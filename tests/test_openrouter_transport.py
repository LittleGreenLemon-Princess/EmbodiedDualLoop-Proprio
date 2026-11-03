import io
import json
from pathlib import Path
import ssl
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import Mock, patch

from orchestrator import openrouter_agent_host as host
from orchestrator.dual_loop.policies import OpenRouterPolicy
from orchestrator.openrouter_transport import (
    DEADLINE_ENV, TransportDeadlineExceeded, deadline_environment,
    retry_after_seconds, session_deadline)


class Clock:
    def __init__(self):
        self.value = 0.
        self.waits = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


class Opener:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.data, timeout))
        outcome = next(self.outcomes)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())


def response(finish='stop', calls=None):
    return {'id': 'test-receipt', 'usage': {'cost': .01}, 'choices': [{
        'finish_reason': finish, 'message': {'role': 'assistant', 'content': 'done',
                                            **({'tool_calls': calls} if calls else {})}}]}


class TransportTests(unittest.TestCase):
    def request(self, outcomes, deadline=600):
        clock, events, opener = Clock(), [], Opener(outcomes)
        return clock, events, opener, lambda: host.api_request('synthetic-test-key', opener, {'test': True},
            lambda kind, payload: events.append({'kind': kind, **payload}), deadline=deadline,
            clock=clock, sleep=clock.sleep, wall_time=lambda: 0.)

    def test_five_eof_failures_recover_without_changing_payload(self):
        clock, events, opener, run = self.request([urllib.error.URLError(ssl.SSLEOFError(8, 'EOF'))] * 5 + [response()])
        self.assertEqual(run()['id'], 'test-receipt')
        self.assertEqual(clock.waits, [10, 20, 40, 60, 60])
        self.assertEqual(len(opener.calls), 6)
        self.assertEqual(len({wire for wire, _ in opener.calls}), 1)
        self.assertEqual(opener.calls[-1][1], 410)
        self.assertEqual(len([e for e in events if e['kind'] == 'api_transport_retry']), 5)

    def test_certificate_auth_and_local_errors_are_terminal(self):
        certificate = ssl.SSLCertVerificationError(1, 'certificate verify failed')
        for error in [certificate, urllib.error.URLError(certificate),
                      urllib.error.HTTPError('https://example.invalid', 401, 'unauthorized', {}, io.BytesIO()),
                      FileNotFoundError('missing config')]:
            with self.subTest(error=error):
                clock, events, opener, run = self.request([error])
                with self.assertRaises(type(error)):
                    run()
                self.assertEqual(len(opener.calls), 1)
                self.assertEqual(clock.waits, [])
                self.assertFalse(events[-1]['retry'])
                self.assertTrue(events[-1]['terminal'])

    def test_eighth_failure_has_terminal_record_and_no_extra_sleep(self):
        clock, events, opener, run = self.request([ConnectionResetError('reset')] * 8)
        with self.assertRaises(ConnectionResetError):
            run()
        self.assertEqual(len(opener.calls), 8)
        self.assertEqual(len(clock.waits), 7)
        self.assertEqual(events[-1]['attempt'], 8)
        self.assertEqual(events[-1]['stop_reason'], 'attempt_limit')
        self.assertFalse(events[-1]['retry'])
        self.assertLess(clock(), 600)

    def test_retry_after_seconds_and_date_and_too_late_retry(self):
        for header in ['45', 'Thu, 01 Jan 1970 00:00:45 GMT']:
            error = urllib.error.HTTPError('https://example.invalid', 429, 'wait', {'Retry-After': header}, io.BytesIO())
            clock, events, opener, run = self.request([error, response()])
            run()
            self.assertEqual(clock.waits, [45])
        self.assertIsNone(retry_after_seconds({'Retry-After': 'nonsense'}))
        error = urllib.error.HTTPError('https://example.invalid', 503, 'wait', {'Retry-After': '60'}, io.BytesIO())
        clock, events, opener, run = self.request([error], deadline=30)
        with self.assertRaises(urllib.error.HTTPError):
            run()
        self.assertEqual(clock.waits, [])
        self.assertEqual(events[-1]['stop_reason'], 'recovery_deadline')

    def test_late_receipt_is_retained_but_cannot_return_tools(self):
        clock, events, opener, run = self.request([], deadline=10)
        def late():
            clock.value = 11
            return response('tool_calls', [{'id': 'a', 'function': {'name': 'shell', 'arguments': '{}'}}])
        opener.outcomes = iter([late])
        with self.assertRaises(TransportDeadlineExceeded):
            run()
        self.assertEqual(opener.calls[0][1], 10)
        self.assertTrue(any(e['kind'] == 'api_receipt' and e['usage']['cost'] == .01 for e in events))
        self.assertFalse(events[-1]['retry'])

    def test_partial_or_error_response_tools_are_never_applied(self):
        for finish in ['error', 'content_filter', None, 'unknown', 'stop']:
            with self.subTest(finish=finish):
                with self.assertRaises(ValueError):
                    host.complete_choice(response(finish, [{'id': 'partial'}]))

    def test_recovered_response_tool_runs_once_in_real_agent_loop(self):
        clock = Clock()
        tool = {'id': 'call-1', 'type': 'function', 'function': {'name': 'shell', 'arguments': '{"command":"inspect only"}'}}
        opener = Opener([ssl.SSLEOFError(8, 'EOF')] * 5 + [response('tool_calls', [tool]), response()])
        original = host.api_request
        def call(key, opener, payload, log, **kwargs):
            return original(key, opener, payload, log, clock=clock, sleep=clock.sleep, wall_time=lambda: 0., **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode/'workspace').mkdir()
            (episode/'workspace/instruction.md').write_text('CPU fixture', encoding='utf-8')
            args = SimpleNamespace(episode=episode, log_dir=episode/'agent', role='solve', phase='playground',
                                   container='fake', container_socket='fake', image='fake', model='pinned', effort='high', max_rounds=2)
            with patch.object(host, 'read_key', return_value='synthetic-test-key'), patch.object(host, 'openrouter_opener', return_value=opener), \
                 patch.object(host, 'session_deadline', return_value=None), patch.object(host, 'api_request', side_effect=call), \
                 patch.object(host, 'execute_tool', return_value=({'exit_code': 0}, [])) as tool_run, patch.object(host.signal, 'signal'):
                self.assertEqual(host.run(args), 0)
            tool_run.assert_called_once()
            self.assertEqual(len(opener.calls), 7)

    def test_s1_retries_share_one_timeout_and_certificate_does_not_retry(self):
        clock = Clock()
        policy = OpenRouterPolicy(timeout=120, transport_retries=1)
        timeouts = []
        def transport(_payload):
            timeouts.append(policy._request_timeout)
            clock.value += 70
            raise ConnectionResetError('reset')
        policy._transport = transport
        with patch('orchestrator.dual_loop.policies.time.monotonic', clock), patch('orchestrator.dual_loop.policies.time.sleep', clock.sleep), \
             patch('orchestrator.openrouter_transport.time.monotonic', clock), self.assertRaises(ConnectionResetError):
            policy.select({}, {'advance': {}})
        self.assertEqual(timeouts, [120, 49.75])
        certificate = urllib.error.URLError(ssl.SSLCertVerificationError(1, 'certificate verify failed'))
        policy._transport = Mock(side_effect=certificate)
        with self.assertRaises(urllib.error.URLError):
            policy.select({}, {'advance': {}})
        self.assertEqual(policy._transport.call_count, 1)

    def test_s1_late_paid_receipt_cannot_become_decision(self):
        clock = Clock()
        policy = OpenRouterPolicy(timeout=120)
        policy.deadline = 10.
        def late(_payload):
            clock.value = 11.
            return response(), 11.
        policy._transport = late
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory)/'api.jsonl'
            with patch('orchestrator.dual_loop.policies.time.monotonic', clock), \
                 patch.object(policy, '_validate_answer') as validate, self.assertRaises(TransportDeadlineExceeded):
                policy.select({}, {'advance': {}})
            validate.assert_not_called()
            events = [json.loads(line) for line in policy.audit_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events[0]['kind'], 'api_receipt')
            self.assertEqual(events[0]['usage']['cost'], .01)
            self.assertTrue(events[-1]['terminal'])

    def test_host_deadline_is_inherited_without_extension(self):
        with patch.dict('os.environ', {DEADLINE_ENV: '25.0'}):
            self.assertEqual(session_deadline(), 25.)
            self.assertEqual(float(deadline_environment(60.)[DEADLINE_ENV]), 25.)
            self.assertEqual(float(deadline_environment(10.)[DEADLINE_ENV]), 10.)
        clock, events, opener, run = self.request([], deadline=0)
        with self.assertRaises(TransportDeadlineExceeded):
            run()
        self.assertEqual(opener.calls, [])
        self.assertEqual(events[-1]['stop_reason'], 'recovery_deadline')


if __name__ == '__main__':
    unittest.main()
