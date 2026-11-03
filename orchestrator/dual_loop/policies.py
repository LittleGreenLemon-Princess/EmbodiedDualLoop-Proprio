"""Replaceable selectors. API credentials stay in the host process only."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.request

from .model import Decision
from orchestrator.openrouter_transport import error_metadata, remaining, retry_after_seconds, retryable


class SelectorConfigurationError(RuntimeError):
    """Host selector configuration is missing; no model decision was made."""


class OpenRouterCompletionLimitError(ValueError):
    """A paid response reached the configured completion limit; no action selected."""


class OpenRouterIncompleteResponseError(ValueError):
    """A response ended without a complete decision; no action selected."""


class AdvancePolicy:
    """A deliberately simple control-flow baseline, not a learned controller."""

    def select(self, state, options):
        if 'advance' in options:
            return Decision('advance')
        if 'escalate' in options:
            return Decision('escalate', judgement='stuck')
        raise ValueError('Baseline needs an advance or escalate option')


class JevPolicy:
    endpoint = 'https://api.typesafe.ai/v1/systemone'

    def __init__(self, *, model='jev-1.13.0', timeout=1., transport=None):
        if not isinstance(model, str) or not model:
            raise ValueError('A pinned Jev model ID is required')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('timeout must be positive and finite')
        self.model, self.timeout = model, timeout
        self._transport = transport or self._request
        # A private host setting scopes proxying to this client, not the simulator
        # or other providers. An empty setting explicitly uses a direct connection.
        proxy = os.environ.get('TYPESAFE_PROXY_URL', '').strip()
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({'https': proxy} if proxy else {}))

    def _request(self, payload):
        key = os.environ.get('TYPESAFE_API_KEY')
        if not key:
            raise SelectorConfigurationError('TYPESAFE_API_KEY is required on the host')
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            method='POST')
        # No automatic retries; slow or failed decisions return control to System 2.
        audit = getattr(self, 'audit_path', None)
        started = time.monotonic()
        if audit:
            from orchestrator.files import append_jsonl
            append_jsonl(audit, {'kind': 'request', 'payload': payload})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                answer = json.load(response)
        except Exception as error:
            if audit:
                append_jsonl(audit, {'kind': 'error', 'error_type': type(error).__name__,
                                    'http_status': getattr(error, 'code', None),
                                    'seconds': time.monotonic() - started})
            raise
        if audit:
            append_jsonl(audit, {'kind': 'response', 'payload': answer,
                                'seconds': time.monotonic() - started})
        return answer

    def select(self, state, options):
        if getattr(self, 'canonical_pose_menu', False):
            menu = {'advance': (
                        'Execute the supplied pose servo code for one waypoint: translate toward '
                        'target XYZ by at most 10 mm AND rotate toward target quaternion by at '
                        'most 0.10 rad using quaternion interpolation. It corrects attitude even '
                        'when position is already within tolerance. Preserve both grippers and '
                        'the other arm.'),
                    'escalate': 'Stop this subtask and request System 2 assistance.'}
            if not set(options) <= set(menu):
                raise ValueError('Canonical servo menu supports only advance and escalate')
            options = {key: menu[key] for key in options}
        payload = {
            'model': self.model, 'state': state,
            'questions': {
                'next_action': {
                    'type': 'choice',
                    'instructions': (
                        'Choose one available action for the current subtask only. '
                        'Use the measured evidence and recent action effects. '
                        'Do not do arithmetic or infer unseen physical success. '
                        'Choose escalate if the available actions cannot help.'),
                    'criteria': options,
                },
                'current_status': {
                    'type': 'choice',
                    'instructions': 'Judge the CURRENT observation, before the next action.',
                    'criteria': {
                        'continue': 'The current subtask is not complete and useful actions remain.',
                        'done': 'Observed physical evidence establishes the specified subtask goal.',
                        'stuck': 'The current approach has failed or useful progress has stopped.',
                        'unknown': 'Evidence is insufficient to decide or act reliably.',
                    },
                },
            },
        }
        response = self._transport(payload)
        answers = response['answers']
        action = answers['next_action']
        status = answers['current_status']
        self._validate_distribution(action, set(options))
        self._validate_distribution(status, {'continue', 'done', 'stuck', 'unknown'})
        # A confident action is unusable when the model is unsure of the situation.
        decision = Decision(action['choice'], min(action['confidence'], status['confidence']),
                            status['choice'], response.get('model', self.model),
                            action['probabilities'])
        decision.validate(options)
        return decision

    @staticmethod
    def _validate_distribution(answer, expected):
        probabilities = answer['probabilities']
        if (answer.get('type') != 'choice' or answer.get('choice') not in expected
                or set(probabilities) != expected):
            raise ValueError('Malformed Jev choice')
        values = list(probabilities.values())
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError('Malformed Jev probabilities')
        # The live API returns probabilities rounded to hundredths. Retain
        # those raw values, allowing at most half a hundredth per option
        # (capped at 5%); still reject materially invalid distributions.
        rounding_tolerance = min(.05, .005 * len(values)) + 1e-9
        if abs(sum(values) - 1) > rounding_tolerance:
            raise ValueError('Jev probabilities must sum to one')
        confidence = answer['confidence']
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (
                math.isfinite(confidence) and 0 <= confidence <= 1):
            raise ValueError('Malformed Jev confidence')


class JevFactorizedPolicy(JevPolicy):
    """Known-target free-space hover only: select once, geometrically check distance."""

    def select(self, state, options):
        pairs = {}
        allowed_directions = {axis + '_' + sign for axis in ('x', 'y', 'z', 'rx', 'ry', 'rz')
                              for sign in ('positive', 'negative')}
        for option in options:
            if option == 'escalate':
                continue
            direction, size = option.split('@')
            if (direction not in allowed_directions or not math.isfinite(float(size))
                    or not 0 < float(size) <= (5 if direction.startswith('r') else 6)):
                raise ValueError('Invalid bounded direction/distance option')
            pairs[option] = (direction, size)
        if not pairs:
            if 'escalate' not in options:
                raise ValueError('No factorized options')
            return Decision('escalate', judgement='stuck', model=self.model)
        directions = {direction for direction, _ in pairs.values()}
        groups = {}
        for d, size in pairs.values():
            groups.setdefault('step_deg' if d.startswith('r') else 'step_mm', set()).add(size)
        for group, sizes in groups.items():
            ds = {d for d in directions if d.startswith('r') == (group == 'step_deg')}
            if sum((d, size) in pairs.values() for d in ds for size in sizes) != len(ds) * len(sizes):
                raise ValueError('Factorized options must provide all allowed combinations per unit')
        direction_criteria = {}
        for d in sorted(directions):
            axis, sign = d.split('_')
            if axis.startswith('r'):
                direction_criteria[d] = (
                    'Apply a ' + sign.upper() + ' right-hand rotation about WORLD '
                    + axis[1:].upper() + ': q_next = q_delta(' +
                    ('+' if sign == 'positive' else '-') + 'step_deg) * q_measured. '
                    'This reduces a ' + sign.upper() + ' component of the REQUIRED '
                    'rotation vector target_rotation_error_world_deg. The vector is '
                    'the command still required, NOT a measured deviation to negate. '
                    'Hold measured XYZ and both grippers.')
            else:
                direction_criteria[d] = ('Translate along world ' + d.replace('_', ' ')
                                        + '; preserve orientation and grippers.')
        if 'escalate' in options:
            direction_criteria['escalate'] = 'Stop this subtask and request System 2 assistance.'
        payload = {'model': self.model, 'state': state, 'questions': {
            'direction': {'type': 'choice', 'instructions': (
                'Choose the movement direction only, independently of step size. '
                'target_minus_measured_mm is target minus measured position in world x,y,z. '
                'A negative component calls for negative motion on that axis; positive calls '
                'for positive motion. For rx/ry/rz use target_rotation_error_world_deg: '
                'the world-axis rotation vector of target * inverse(measured), not Euler-angle '
                'subtraction. This vector is the REQUIRED signed correction: positive rx/ry/rz '
                'requires positive rotation, negative requires negative rotation. DO NOT negate '
                'it again. A +6 degree required X rotation calls for rx_positive, and a -6 '
                'degree required X rotation calls for rx_negative; the SAME convention applies '
                'to Y and Z. These are static coordinate definitions, not an answer for this state. '
                'Select translation when positional error needs correction and '
                'rotation when attitude error needs correction; compare each error with its '
                'own tolerance, never mix mm and degrees. Use current evidence and recent effects. '
                'Choose escalate if the available movements cannot help.'),
                'criteria': direction_criteria},
            'current_status': {'type': 'choice',
                'instructions': 'Judge the CURRENT observation, before the next action.',
                'criteria': {
                    'continue': 'The goal is not yet reached and useful actions remain.',
                    'done': 'Measured evidence establishes ALL stated position and attitude tolerances.',
                    'stuck': 'Useful progress has stopped.',
                    'unknown': 'Evidence is insufficient to act reliably.'}}}}
        for key, sizes in groups.items():
            unit = 'degrees' if key == 'step_deg' else 'mm'
            error_field = ('target_rotation_error_world_deg' if key == 'step_deg'
                           else 'target_minus_measured_mm')
            payload['questions'][key] = {'type': 'choice', 'instructions': (
                'Choose the size of one correction in ' + unit + ', independently of direction. '
                'Use the dominant component of ' + error_field + ' and the corresponding tolerance. '
                'Choose afresh each tick. Prefer larger useful corrections far from target '
                'and smaller corrections near it, without exceeding the remaining component. '
                'Only the size matching the selected motion type will be executed.'),
                'criteria': {size: 'Apply ' + size + ' ' + unit + ' this tick.'
                             for size in sorted(sizes, key=float)}}
        response = self._transport(payload)
        answers = response['answers']
        for key, criteria in [('direction', direction_criteria), *groups.items(),
                              ('current_status', {'continue', 'done', 'stuck', 'unknown'})]:
            self._validate_distribution(answers[key], set(criteria))
        direction, status = (answers[k] for k in ('direction', 'current_status'))
        rotational = direction['choice'].startswith('r')
        size_key = 'step_deg' if rotational else 'step_mm'
        size = answers.get(size_key)
        action = ('escalate' if direction['choice'] == 'escalate' else
                  direction['choice'] + '@' + size['choice'])
        details = {'confidence_by_question': {k: a['confidence'] for k, a in answers.items()},
                   'proposed_action': action, 'distance_check': 'not_a_movement'}
        if action != 'escalate' and status['choice'] == 'continue':
            current = state['current']
            delta = current['target_rotation_error_world_deg' if rotational
                            else 'target_minus_measured_mm']
            if (current.get('motion_mode') != 'free_space_hover' or len(delta) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in delta)):
                raise ValueError('Distance check requires finite measured free-space hover error')
            axis, sign = direction['choice'].split('_')
            if rotational:
                axis = axis[1:]
            remaining = delta['xyz'.index(axis)] * (1 if sign == 'positive' else -1)
            distance_ok = 0 < float(size['choice']) <= remaining + 1e-9
            details['distance_check'] = 'passed' if distance_ok else 'rejected'
            details['remaining_in_selected_direction_mm'] = remaining
            if rotational:
                details['remaining_in_selected_direction_deg'] = details.pop(
                    'remaining_in_selected_direction_mm')
            if not distance_ok:
                action = 'escalate'
        # Several distances can be useful. Preserve the model's uncertainty,
        # but gate movement on direction/status confidence and a non-overshoot
        # geometry check, rather than requiring one distance to dominate.
        decision = Decision(action, min(direction['confidence'], status['confidence']),
                            status['choice'], response.get('model', self.model),
                            {key: answers[key]['probabilities'] for key in answers}, details)
        decision.validate(options)
        return decision


class OpenRouterHTTPError(RuntimeError):
    def __init__(self, status_code, detail, retry_after=None):
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f'OpenRouter HTTP {status_code}: {detail}')


class OpenRouterPolicy:
    """Explicit noncanonical System 1 selector backed by OpenRouter chat completions.

    Credentials are read by this host process from an external auth file. The
    workspace-only shell container never receives them. This is an adaptation
    condition: it preserves the dual-loop executor but does not call Jev.
    """

    endpoint = 'https://openrouter.ai/api/v1/chat/completions'
    max_completion_tokens = 16384

    def __init__(self, *, model='deepseek/deepseek-v4.1-flash', effort='high',
                 timeout=120., transport=None, transport_retries=1):
        if not isinstance(model, str) or not model:
            raise ValueError('A pinned OpenRouter model ID is required')
        if effort not in ('low', 'medium', 'high', 'xhigh'):
            raise ValueError('Unsupported reasoning effort')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('timeout must be positive and finite')
        if (isinstance(transport_retries, bool) or not isinstance(transport_retries, int)
                or not 0 <= transport_retries <= 1):
            raise ValueError('transport_retries must be an integer from 0 to 1')
        self.model, self.effort, self.timeout = model, effort, timeout
        self.transport_retries = transport_retries
        self._transport = transport or self._request
        proxy = (os.environ.get('EMBODIEDEVO_OPENROUTER_PROXY_URL')
                 or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '').strip()
        proxies = {'https': proxy, 'http': proxy} if proxy else {}
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))

    @staticmethod
    def _key():
        path = os.environ.get('EMBODIEDEVO_OPENROUTER_AUTH_FILE')
        if not path:
            raise SelectorConfigurationError('EMBODIEDEVO_OPENROUTER_AUTH_FILE is required on the host')
        with open(path, encoding='utf-8-sig') as stream:
            auth = json.load(stream)
        key = auth.get('OPENAI_API_KEY')
        if not isinstance(key, str) or not key:
            raise SelectorConfigurationError('OpenRouter auth file is missing OPENAI_API_KEY')
        return key

    def _request(self, payload):
        key = self._key()
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
            method='POST')
        started = time.monotonic()
        try:
            with self._opener.open(request, timeout=getattr(self, '_request_timeout', self.timeout)) as response:
                return json.load(response), time.monotonic() - started
        except urllib.error.HTTPError as error:
            # Read but never persist a provider body that might contain secrets.
            detail = error.read(2048).decode('utf-8', errors='replace').replace(key, '[redacted]')
            raise OpenRouterHTTPError(error.code, detail[:1000], retry_after_seconds(error.headers)) from None

    @staticmethod
    def _retryable_transport_error(error):
        return retryable(error)

    @staticmethod
    def _content(message):
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
        raise ValueError('OpenRouter returned no structured text content')

    @staticmethod
    def _validate_answer(answer, options):
        expected = set(options)
        action = answer.get('next_action')
        status = answer.get('current_status')
        probs = answer.get('action_probabilities')
        if action not in expected or status not in {'continue', 'done', 'stuck', 'unknown'}:
            raise ValueError('OpenRouter returned an unavailable choice')
        if not isinstance(probs, dict) or set(probs) != expected:
            raise ValueError('OpenRouter returned a malformed action distribution')
        values = list(probs.values())
        if any(isinstance(p, bool) or not isinstance(p, (int, float))
               or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError('OpenRouter returned invalid action probabilities')
        probability_sum = sum(values)
        if probability_sum <= 0:
            raise ValueError('OpenRouter action probabilities must have positive mass')
        # These are uncalibrated model-reported weights, not a safety signal.
        # Normalize any finite nonnegative vector with positive mass, while
        # retaining the raw total in the audit record.
        normalized_probs = {key: value / probability_sum for key, value in probs.items()}
        for name in ('action_confidence', 'status_confidence'):
            value = answer.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('OpenRouter returned invalid self-reported confidence')
        rationale = answer.get('rationale', '')
        if not isinstance(rationale, str) or len(rationale) > 1200:
            raise ValueError('OpenRouter returned an invalid rationale')
        return action, status, normalized_probs, rationale, probability_sum

    def select(self, state, options):
        if not isinstance(state, dict) or not isinstance(options, dict) or not options:
            raise ValueError('System 1 requires public state and a nonempty legal option menu')
        keys = sorted(options)
        schema = {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'next_action': {'type': 'string', 'enum': keys},
                'action_probabilities': {'type': 'object', 'additionalProperties': False,
                    'properties': {key: {'type': 'number', 'minimum': 0, 'maximum': 1}
                                   for key in keys}, 'required': keys},
                'current_status': {'type': 'string',
                    'enum': ['continue', 'done', 'stuck', 'unknown']},
                'action_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'status_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'rationale': {'type': 'string', 'maxLength': 1200},
            },
            'required': ['next_action', 'action_probabilities', 'current_status',
                         'action_confidence', 'status_confidence', 'rationale'],
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': (
                    'You are System 1 in an experimental dual-loop robot controller. '
                    'Select exactly one legal option from the supplied menu using only '
                    'the public measured state, goal, and recent effects. Never invent an '
                    'option, infer unseen success, or direct a different action. Judge status '
                    'before the selected action. If the state identifies an active stage and a '
                    'legal bounded action can make progress, choose continue; do not choose '
                    'unknown merely because completion is not yet verified. Reserve unknown '
                    'for missing or contradictory goal/measurement evidence. Choose done only '
                    'when the supplied physical evidence meets the stated completion criterion. '
                    'Confidence values and probabilities are '
                    'your self-reports, not calibrated measurements. Return valid JSON only.')},
                {'role': 'user', 'content': json.dumps(
                    {'state': state, 'available_options': options},
                    ensure_ascii=False, allow_nan=False, separators=(',', ':'))},
            ],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'system1_decision', 'strict': True, 'schema': schema}},
            'max_tokens': self.max_completion_tokens, 'temperature': 0,
            'reasoning': {'effort': self.effort}, 'stream': False,
        }
        request_sha = hashlib.sha256(json.dumps(
            payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        transport_attempts = 0
        started = time.monotonic()
        deadline = min(started + self.timeout, getattr(self, 'deadline', float('inf')))
        request_bytes = len(json.dumps(payload, allow_nan=False).encode())
        try:
            while True:
                self._request_timeout = remaining(deadline)
                transport_attempts += 1
                try:
                    response, elapsed = self._transport(payload)
                    break
                except Exception as error:
                    after = getattr(error, 'retry_after', None)
                    if after is None:
                        after = retry_after_seconds(getattr(error, 'headers', None))
                    delay = max(min(.25 * (2 ** (transport_attempts - 1)), 1.0), after or 0.)
                    if (transport_attempts > self.transport_retries
                            or not self._retryable_transport_error(error)
                            or time.monotonic() + delay >= deadline):
                        raise
                    if getattr(self, 'audit_path', None):
                        from orchestrator.files import append_jsonl
                        append_jsonl(self.audit_path, {
                            'kind': 'transport_retry', 'request_sha256': request_sha,
                            'attempt': transport_attempts,
                            **error_metadata(error), 'request_bytes': request_bytes,
                            'retry_after_seconds': after, 'backoff_seconds': delay,
                            'seconds': time.monotonic() - started,
                            'retry': True, 'robot_action_applied': False,
                        })
                    time.sleep(delay)
            choices = response.get('choices') or []
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            receipt = {'kind': 'api_receipt', 'request_sha256': request_sha,
                       'id': response.get('id'), 'model': response.get('model'),
                       'provider': response.get('provider'), 'usage': response.get('usage'),
                       'finish_reason': choice.get('finish_reason'), 'seconds': elapsed,
                       'transport_attempts': transport_attempts, 'request_bytes': request_bytes}
            if getattr(self, 'audit_path', None):
                from orchestrator.files import append_jsonl
                append_jsonl(self.audit_path, receipt)
            remaining(deadline)
            if response.get('error') or choice.get('error'):
                raise OpenRouterIncompleteResponseError('OpenRouter returned a provider error; no action selected')
            if choice.get('finish_reason') == 'length':
                raise OpenRouterCompletionLimitError(
                    'OpenRouter System 1 reached its configured completion limit')
            if choice.get('finish_reason') not in ('stop', 'end_turn'):
                raise OpenRouterIncompleteResponseError('OpenRouter System 1 response was incomplete')
            text = self._content(choice['message'])
            answer = json.loads(text)
            action, status, probabilities, rationale, probability_sum = self._validate_answer(
                answer, options)
            if getattr(self, 'audit_path', None):
                from orchestrator.files import append_jsonl
                append_jsonl(self.audit_path, {'kind': 'validated_decision',
                                               'request_sha256': request_sha, 'answer': answer,
                                               'raw_action_probability_sum': probability_sum,
                                               'probabilities_normalized': abs(probability_sum - 1) > 1e-9})
            decision = Decision(action, min(answer['action_confidence'],
                                            answer['status_confidence']), status,
                                response.get('model', self.model), probabilities,
                                {'rationale': rationale,
                                 'action_confidence': answer['action_confidence'],
                                 'status_confidence': answer['status_confidence'],
                                 'raw_action_probability_sum': probability_sum,
                                 'provider': response.get('provider')})
            decision.validate(options)
            return decision
        except Exception as error:
            if getattr(self, 'audit_path', None):
                from orchestrator.files import append_jsonl
                append_jsonl(self.audit_path, {
                    'kind': 'error', 'request_sha256': request_sha,
                    **error_metadata(error), 'message': str(error)[:1200],
                    'retry': False, 'attempts': transport_attempts,
                    'retries_used': max(0, transport_attempts - 1),
                    'terminal': True, 'elapsed_seconds': time.monotonic() - started,
                    'request_bytes': request_bytes,
                    'robot_action_applied': False})
            raise


def make_policy(settings):
    if not isinstance(settings, dict):
        raise ValueError('Selector settings must be an object')
    kind = settings.get('kind', 'advance')
    if kind == 'advance':
        return AdvancePolicy()
    if kind in ('jev', 'jev_factorized'):
        policy_class = JevPolicy if kind == 'jev' else JevFactorizedPolicy
        return policy_class(model=settings.get('model', 'jev-1.13.0'),
                            timeout=settings.get('timeout', 1.))
    if kind == 'openrouter':
        return OpenRouterPolicy(model=settings.get('model', 'deepseek/deepseek-v4.1-flash'),
                                effort=settings.get('effort', 'high'),
                                timeout=settings.get('timeout', 120.),
                                transport_retries=settings.get('transport_retries', 1))
    raise ValueError('Supported selectors: advance, jev, jev_factorized, openrouter')
