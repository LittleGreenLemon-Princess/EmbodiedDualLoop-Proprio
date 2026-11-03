import json
import tempfile
import unittest
from pathlib import Path
import urllib.error
from unittest.mock import Mock, patch

from orchestrator.dual_loop.policies import (
    OpenRouterPolicy, OpenRouterCompletionLimitError, OpenRouterIncompleteResponseError, make_policy)


class OpenRouterPolicyTests(unittest.TestCase):
    def test_structured_selection_uses_only_current_legal_menu(self):
        captured = {}
        answer = {
            'next_action': 'place_red',
            'action_probabilities': {'place_red': .8, 'inspect': .2},
            'current_status': 'continue',
            'action_confidence': .9,
            'status_confidence': .95,
            'rationale': 'Current public evidence supports the placement option.',
        }

        def transport(payload):
            captured.update(payload)
            return ({'id': 'receipt-1', 'model': 'deepseek/deepseek-v4.1-flash',
                     'provider': 'test-provider', 'usage': {'total_tokens': 12},
                     'choices': [{'finish_reason': 'stop', 'message': {
                         'content': json.dumps(answer)}}]}, 0.25)

        policy = OpenRouterPolicy(transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory) / 'system1.jsonl'
            decision = policy.select({'goal': 'place red block', 'current': {'stage': 2}},
                                     {'place_red': {'code': 'step(...)'},
                                      'inspect': {'code': 'no action'}})
            records = [json.loads(line) for line in policy.audit_path.read_text().splitlines()]
        self.assertEqual(decision.action, 'place_red')
        self.assertEqual(decision.model, 'deepseek/deepseek-v4.1-flash')
        self.assertEqual(set(decision.probabilities), {'place_red', 'inspect'})
        self.assertEqual(captured['reasoning']['effort'], 'high')
        self.assertEqual(captured['max_tokens'], 16384)
        self.assertEqual(captured['response_format']['type'], 'json_schema')
        self.assertIn('do not choose unknown merely because completion is not yet verified',
                      captured['messages'][0]['content'].lower())
        self.assertEqual([row['kind'] for row in records],
                         ['api_receipt', 'validated_decision'])
        self.assertEqual(records[0]['id'], 'receipt-1')

    def test_rejects_unavailable_choice_and_still_records_paid_receipt(self):
        response = {
            'id': 'receipt-invalid', 'model': 'deepseek/deepseek-v4.1-flash',
            'usage': {'total_tokens': 8},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                'next_action': 'invented', 'action_probabilities': {'legal': 1},
                'current_status': 'continue', 'action_confidence': .9,
                'status_confidence': .9, 'rationale': 'invalid choice'})}}],
        }
        policy = OpenRouterPolicy(transport=lambda _payload: (response, 0.1))
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory) / 'system1.jsonl'
            with self.assertRaises(ValueError):
                policy.select({'goal': 'test'}, {'legal': {'code': 'step(...)'}})
            records = [json.loads(line) for line in policy.audit_path.read_text().splitlines()]
        self.assertEqual(records[0]['kind'], 'api_receipt')
        self.assertEqual(records[0]['id'], 'receipt-invalid')
        self.assertEqual(records[-1]['kind'], 'error')
        self.assertFalse(records[-1]['robot_action_applied'])

    def test_normalizes_positive_action_probability_weights(self):
        answer = {
            'next_action': 'place_red',
            'action_probabilities': {'place_red': .6, 'inspect': .3},
            'current_status': 'continue',
            'action_confidence': .9,
            'status_confidence': .9,
            'rationale': 'The legal action is supported by the current evidence.',
        }
        response = {'id': 'receipt-rounded', 'model': 'deepseek/deepseek-v4.1-flash',
                    'usage': {'total_tokens': 8},
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'content': json.dumps(answer)}}]}
        policy = OpenRouterPolicy(transport=lambda _payload: (response, .1))
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory) / 'system1.jsonl'
            decision = policy.select({'goal': 'place red block'},
                                     {'place_red': {'code': 'step(...)'},
                                      'inspect': {'code': 'no action'}})
            records = [json.loads(line) for line in policy.audit_path.read_text().splitlines()]
        self.assertEqual(decision.action, 'place_red')
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.)
        self.assertAlmostEqual(decision.probabilities['place_red'], 2 / 3)
        validated = next(row for row in records if row['kind'] == 'validated_decision')
        self.assertAlmostEqual(validated['raw_action_probability_sum'], .9)
        self.assertTrue(validated['probabilities_normalized'])

    def test_retries_transient_transport_error_before_returning_a_decision(self):
        answer = {
            'next_action': 'place_red',
            'action_probabilities': {'place_red': .8, 'inspect': .2},
            'current_status': 'continue',
            'action_confidence': .9,
            'status_confidence': .9,
            'rationale': 'The measured state supports this legal action.',
        }
        response = {'id': 'receipt-after-retry', 'model': 'deepseek/deepseek-v4.1-flash',
                    'usage': {'total_tokens': 8},
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'content': json.dumps(answer)}}]}
        calls = []

        def transport(_payload):
            calls.append(True)
            if len(calls) == 1:
                raise urllib.error.URLError('temporary TLS EOF')
            return response, .2

        policy = OpenRouterPolicy(transport=transport, transport_retries=1)
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory) / 'system1.jsonl'
            decision = policy.select({'goal': 'place red block'},
                                     {'place_red': {'code': 'step(...)'},
                                      'inspect': {'code': 'no action'}})
            records = [json.loads(line) for line in policy.audit_path.read_text().splitlines()]
        self.assertEqual(decision.action, 'place_red')
        self.assertEqual(len(calls), 2)
        self.assertEqual([row['kind'] for row in records],
                         ['transport_retry', 'api_receipt', 'validated_decision'])
        self.assertFalse(records[0]['robot_action_applied'])
        self.assertEqual(records[1]['transport_attempts'], 2)

    def test_model_policy_is_explicit_and_validated_without_api_call(self):
        policy = make_policy({'kind': 'openrouter', 'model': 'deepseek/deepseek-v4.1-flash',
                              'effort': 'high', 'timeout': 120})
        self.assertIsInstance(policy, OpenRouterPolicy)
        with self.assertRaises(ValueError):
            make_policy({'kind': 'openrouter', 'effort': 'maximum'})

    def test_exhausted_transport_retry_is_terminal_and_records_attempts(self):
        transport = Mock(side_effect=urllib.error.URLError('TLS EOF'))
        policy = OpenRouterPolicy(transport=transport, transport_retries=1)
        with tempfile.TemporaryDirectory() as directory:
            policy.audit_path = Path(directory) / 'system1.jsonl'
            with patch('orchestrator.dual_loop.policies.time.sleep'), self.assertRaises(
                    urllib.error.URLError):
                policy.select({'goal': 'test'}, {'advance': {}})
            records = [json.loads(line) for line in policy.audit_path.read_text().splitlines()]
        self.assertEqual(transport.call_count, 2)
        self.assertFalse(records[-1]['retry'])
        self.assertEqual(records[-1]['retries_used'], 1)
        self.assertFalse(records[-1]['robot_action_applied'])

    def test_local_auth_file_error_is_not_retried(self):
        transport = Mock(side_effect=FileNotFoundError('auth file missing'))
        policy = OpenRouterPolicy(transport=transport)
        with self.assertRaises(FileNotFoundError):
            policy.select({'goal': 'test'}, {'advance': {}})
        self.assertEqual(transport.call_count, 1)

    def test_incomplete_paid_response_never_retries_or_validates_partial_action(self):
        for finish, exception in [('length', OpenRouterCompletionLimitError),
                                  ('content_filter', OpenRouterIncompleteResponseError)]:
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as directory:
                response = {'id': 'paid-' + finish, 'usage': {
                    'cost': .01, 'completion_tokens': 16384}, 'choices': [{
                    'finish_reason': finish, 'message': {'content': '{"next_action":"advance"'}}]}
                transport = Mock(return_value=(response, .1))
                policy = OpenRouterPolicy(transport=transport)
                policy.audit_path = Path(directory) / 'system1.jsonl'
                with patch.object(policy, '_validate_answer') as validate, self.assertRaises(exception):
                    policy.select({'goal': 'test'}, {'advance': {}})
                validate.assert_not_called()
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(transport.call_args.args[0]['max_tokens'], 16384)
                records = [json.loads(line) for line in policy.audit_path.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(records[0]['kind'], 'api_receipt')
                self.assertEqual(records[0]['usage']['cost'], .01)
                self.assertEqual(records[-1]['error_type'], exception.__name__)
                self.assertFalse(records[-1]['robot_action_applied'])
                self.assertFalse(records[-1]['retry'])


if __name__ == '__main__':
    unittest.main()
