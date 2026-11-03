"""OpenRouter tool agent with credentials kept outside the isolated shell container."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from orchestrator.openrouter_transport import (
    TransportDeadlineExceeded, error_metadata, remaining, retry_after_seconds,
    retryable, session_deadline)

ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


def function(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}}


TOOLS = [
    function('shell', 'Run a shell command inside the isolated workspace-only Docker container. '
             'Use scripts/env.py for observations, code submissions, status and completion. '
             'A timed-out command may have run; inspect state and do not repeat an uncertain action.',
             {'command': {'type': 'string'}}, ['command']),
    function('view_images', 'Inspect up to two current native camera PNG files from observations/. '
             'Do not inspect raw_cam files or create image copies.',
             {'paths': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                        'maxItems': 2}}, ['paths']),
]


def start_shell_container(workspace, socket_path, *, image, cpus, memory_mb, phase):
    """Start an offline shell container; mount only workspace and the session socket."""
    from orchestrator import launch_agent

    launch_agent.require_image(image)
    workspace, socket_path = Path(workspace).resolve(), Path(socket_path).resolve()
    if not workspace.is_dir() or not socket_path.exists():
        raise ValueError('Workspace and initialized session socket are required')
    name = f'embodied-openrouter-{os.getpid()}-{uuid.uuid4().hex[:12]}'
    owner_pid = os.environ.get('EMBODIEDEVO_AGENT_OWNER_PID', str(os.getpid()))
    command = ['docker', 'run', '--detach', '--rm', '--init', '--read-only',
        '--tmpfs', '/tmp:mode=1777', '--network', 'none', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '512',
        '--memory', f'{memory_mb}m', '--cpus', str(cpus),
        '--label', 'org.embodieddualloop.credentials=absent',
        '--label', f'org.embodieddualloop.owner-pid={owner_pid}', '--name', name,
        '--workdir', '/workspace', '--env', 'HOME=/tmp',
        '--env', 'EMBODIEDEVO_WORKSPACE=/workspace',
        '--env', 'EMBODIEDEVO_SOCKET=/tmp/session.sock',
        '--env', 'LANG=C.UTF-8', '--env', 'LC_ALL=C.UTF-8',
        '--mount', f'type=bind,src={workspace},dst=/workspace',
        '--mount', f'type=bind,src={socket_path},dst=/tmp/session.sock,readonly']
    for component in ('instruction.md', 'primitives', 'scripts', 'observations'):
        path = workspace / component
        if path.exists():
            command += ['--mount', f'type=bind,src={path},dst=/workspace/{component},readonly']
    if phase == 'test':
        for component in ('skills', 'lessons'):
            path = workspace / component
            if path.exists():
                command += ['--mount', f'type=bind,src={path},dst=/workspace/{component},readonly']
    command += ['--entrypoint', '/bin/sleep', image, 'infinity']
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f'Could not start offline agent container: {result.stderr[-1000:]}')
    return name, '/tmp/session.sock'


def stop_shell_container(name):
    if not name or not name.startswith('embodied-openrouter-'):
        raise ValueError('Refusing to remove an unowned agent container')
    subprocess.run(['docker', 'rm', '-f', name], stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=30, check=False)


def read_key():
    path = os.environ.get('EMBODIEDEVO_OPENROUTER_AUTH_FILE')
    if not path:
        raise RuntimeError('EMBODIEDEVO_OPENROUTER_AUTH_FILE is required on the host')
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    key = data.get('OPENAI_API_KEY')
    if not isinstance(key, str) or not key:
        raise RuntimeError('OpenRouter auth file has no OPENAI_API_KEY')
    return key


def openrouter_opener():
    proxy = (os.environ.get('EMBODIEDEVO_OPENROUTER_PROXY_URL')
             or os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '').strip()
    return urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': proxy, 'https': proxy} if proxy else {}))


def api_request(key, opener, payload, log, *, deadline=None,
                clock=time.monotonic, sleep=time.sleep, wall_time=time.time):
    # Transport retries occur only before any tool is applied. Each failed
    # attempt is recorded; returned tool calls are applied exactly once below.
    started = clock()
    deadline = min(started + 600., deadline) if deadline is not None else started + 600.
    wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    request_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    for attempt in range(1, 9):
        attempt_started = clock()
        try:
            timeout = remaining(deadline, now=clock())
            log('api_attempt', {'request_sha256': request_sha, 'attempt': attempt,
                               'request_bytes': len(wire), 'timeout_seconds': timeout,
                               'elapsed_seconds': clock() - started})
            request = urllib.request.Request(ENDPOINT,
                wire,
                {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
                method='POST')
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError('OpenRouter response must be a JSON object')
            choices = result.get('choices')
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            log('api_receipt', {'request_sha256': request_sha,
                'id': result.get('id'), 'model': result.get('model'),
                'provider': result.get('provider'), 'usage': result.get('usage'),
                'seconds': clock() - attempt_started, 'attempt': attempt,
                'elapsed_seconds': clock() - started, 'request_bytes': len(wire),
                'finish_reason': choice.get('finish_reason')})
            # urllib's timeout bounds blocking socket operations, not a trickle
            # response's total wall time. Retain late receipts, never their tools.
            remaining(deadline, now=clock())
            if 'error' in result:
                raise RuntimeError('OpenRouter returned an API error: ' +
                                   str(result['error']).replace(key, '[redacted]'))
            return result
        except Exception as error:
            after = retry_after_seconds(getattr(error, 'headers', None), now=wall_time())
            delay = max(min(60., 10. * 2 ** (attempt - 1)), after or 0.)
            transient = retryable(error)
            retry = transient and attempt < 8 and clock() + delay < deadline
            detail = str(error).replace(key, '[redacted]')[:1000]
            kind = ('api_error' if isinstance(error, urllib.error.HTTPError) else
                    'api_transport_retry' if retry else
                    'api_transport_error' if transient or isinstance(error, TransportDeadlineExceeded) else
                    'api_response_error' if isinstance(error, (ValueError, RuntimeError)) else 'api_client_error')
            log(kind, {'request_sha256': request_sha, 'attempt': attempt,
                **error_metadata(error), 'detail': detail, 'request_bytes': len(wire),
                'attempt_seconds': clock() - attempt_started, 'elapsed_seconds': clock() - started,
                'retry': retry, 'terminal': not retry, 'retry_after_seconds': after,
                'backoff_seconds': delay if retry else None, 'robot_tool_applied': False,
                'stop_reason': None if retry else 'recovery_deadline' if isinstance(error, TransportDeadlineExceeded) else
                               'permanent_error' if not transient else
                               'attempt_limit' if attempt == 8 else 'recovery_deadline'})
            if not retry:
                raise
            sleep(delay)


def assistant_message(value):
    return {k: v for k, v in value.items() if k in {
        'role', 'content', 'tool_calls', 'reasoning', 'reasoning_details'}}


def complete_choice(result):
    """Reject partial/error replies before any shell or robot tool can run."""
    if result.get('error'):
        raise ValueError('OpenRouter returned an error response; no tool applied')
    choices = result.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('OpenRouter must return one complete choice')
    choice = choices[0]
    if choice.get('error') or choice.get('finish_reason') not in {'stop', 'tool_calls', 'length'}:
        raise ValueError('OpenRouter reply is incomplete; no tool applied')
    if not isinstance(choice.get('message'), dict):
        raise ValueError('OpenRouter reply has no message')
    calls = choice['message'].get('tool_calls') or []
    if choice['finish_reason'] == 'stop' and calls:
        raise ValueError('Unexpected tools in a stopped reply; no tool applied')
    if choice['finish_reason'] == 'tool_calls' and not calls:
        raise ValueError('Tool reply has no tool calls')
    return choice


def image_item(root, name):
    path = (root / name).resolve()
    relative = path.relative_to(root.resolve())
    if (len(relative.parts) != 3 or relative.parts[0] != 'observations'
            or not relative.parts[1].isdigit() or not path.name.startswith('current_cam_')
            or path.suffix != '.png'):
        raise ValueError('Only observations/NNNNNN/current_cam_*.png may be viewed')
    raw = path.read_bytes()
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Image is not a PNG')
    metadata = {'path': str(relative), 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw)}
    item = {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(raw).decode(), 'detail': 'high'}}
    return item, metadata


def execute_tool(root, container, container_socket, name, arguments):
    if name == 'shell':
        command = arguments['command']
        result = subprocess.run(['docker', 'exec', '--workdir', '/workspace', container,
                                 'bash', '-lc', command], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=1250, check=False)
        output = result.stdout.decode('utf-8', errors='replace')
        return {'exit_code': result.returncode, 'output': output[:240000],
                'output_truncated': len(output) > 240000}, []
    if name == 'view_images':
        paths = arguments['paths']
        if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
            raise ValueError('View one or two images per call')
        pairs = [image_item(root, path) for path in paths]
        return {'images': [metadata for _, metadata in pairs]}, [item for item, _ in pairs]
    raise ValueError('Unknown tool')


def parallel_tool_rejection_messages(calls):
    """Reject every call in a parallel response without executing any of them."""
    if len(calls) < 2:
        raise ValueError('Expected at least two tool calls')
    content = json.dumps({
        'error': 'No tool in the parallel response was executed. '
                 'Send exactly one tool call in the next response; do not replay actions.'
    })
    return [{'role': 'tool', 'tool_call_id': call['id'], 'content': content}
            for call in calls]


def run(args):
    root = (args.episode / 'workspace').resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript = log_dir / 'openrouter-session.jsonl'
    usage_file = log_dir / 'usage.json'
    totals = {name: 0 for name in ('input_tokens', 'cached_input_tokens',
                                   'output_tokens', 'reasoning_output_tokens', 'total_tokens')}
    receipts = 0
    key = read_key()
    opener = openrouter_opener()

    def log(kind, payload):
        safe = json.dumps(payload, ensure_ascii=False).replace(key, '[redacted]')
        with transcript.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'type': kind, 'timestamp': time.time(),
                                     'payload': json.loads(safe)}, ensure_ascii=False) + '\n')
            stream.flush()

    isolation = {'image': args.image, 'container': args.container, 'network': 'none',
        'workspace_only_mount': True, 'api_key_mounted': False, 'api_key_in_container': False,
        'simulator_sources_mounted': False, 'gpu_access': False,
        'socket_mount': 'session RPC only; controller key excluded',
        'read_only_knowledge': args.phase == 'test'}
    (log_dir / 'container-isolation.json').write_text(
        json.dumps(isolation, indent=2) + '\n', encoding='utf-8')
    if args.role == 'solve':
        system = ('You are an autonomous coding agent solving a physical task through a '
                  'restricted workspace. Read instruction.md and the primitive manual. '
                  'Submit all code only through scripts/env.py exec. Native actions and '
                  'code submissions have separate budgets. Use view_images for current '
                  'native camera input. Preserve task goals and report failures honestly. '
                  'Send at most one tool call per response; never request parallel tool calls. '
                  'Never inspect credentials, host paths, simulator source, or change the API runner.')
        user = (root / 'instruction.md').read_text(encoding='utf-8')
    else:
        system = ('You are System 2 in the experimental dual-loop Test repair role. '
                  'Use only public evidence in the current workspace, inspect the specified '
                  'handoff and current observations, then write exactly one local repair '
                  'proposal under submission/. Do not call reset, delegate, finish, or perform '
                  'movement. The host controller will validate the proposal and owns all later '
                  'System 1 movement. skills/ and lessons/ are frozen read-only. If evidence '
                  'does not justify a repair, write a JSON proposal with stop=true and a concise reason. '
                  'Use the handoff proposal_template and proposal_contract: a repair requires a task '
                  'object with entry naming a saved Python file, not inline code or a diagnostic note. '
                  'Preserve the supplied parameters and limits unless the local repair needs a change '
                  'within the frozen bounds. Omit selector to inherit the frozen provider/model/settings; '
                  'changing selector is forbidden. Never edit skills or lessons.')
        user = ('Repair handoff: ' + args.handoff + '\nRead it and follow the repair protocol in '
                'instruction.md. Do not execute physical actions.')
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    log('backend', {'name': 'openrouter-host-api/offline-workspace-container',
                    'model': args.model, 'reasoning_effort': args.effort,
                    'role': args.role, 'max_rounds': args.max_rounds,
                    'container': args.container, 'credentials_exposed_to_container': False})

    def signal_exit(_signum, _frame):
        raise SystemExit('Agent interrupted by session supervisor')

    signal.signal(signal.SIGTERM, signal_exit)
    parallel_tool_rejections = 0
    deadline = session_deadline()
    try:
        for round_index in range(args.max_rounds):
            # Only the host's session record can close the agent. Native episode
            # termination alone still permits Playground export and finish work.
            try:
                state = json.loads((args.episode / 'session-result.json').read_text(encoding='utf-8'))
            except FileNotFoundError:
                pass
            else:
                if not isinstance(state, dict) or type(state.get('closed')) is not bool:
                    raise ValueError('Authoritative session result must contain a boolean closed field')
                if state['closed'] is True:
                    log('session_closed', {'role': args.role, 'round': round_index,
                                           'reason': state.get('reason'), 'valid': state.get('valid')})
                    return 0
            result = api_request(key, opener, {
                'model': args.model, 'messages': messages, 'tools': TOOLS,
                'parallel_tool_calls': False, 'max_tokens': 16384,
                'reasoning': {'effort': args.effort}, 'stream': False}, log, deadline=deadline)
            usage = result.get('usage') or {}
            for target, value in {
                'input_tokens': usage.get('prompt_tokens', 0),
                'cached_input_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'reasoning_output_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
            }.items():
                totals[target] += value or 0
            receipts += 1
            usage_file.write_text(json.dumps({'provider': 'openrouter', 'model': args.model,
                'receipts': receipts, **totals}, indent=2) + '\n', encoding='utf-8')
            try:
                choice = complete_choice(result)
            except ValueError as error:
                log('response_rejected', {'error_type': type(error).__name__,
                                          'reason': str(error), 'robot_tool_applied': False})
                raise
            message = assistant_message(choice['message'])
            log('assistant', message)
            if choice.get('finish_reason') == 'length':
                messages.append({k: v for k, v in message.items() if k != 'tool_calls'})
                log('truncated_response', {'round': round_index, 'robot_tool_applied': False,
                                           'discarded_tool_calls': len(message.get('tool_calls') or [])})
                messages.append({'role': 'user', 'content':
                    'The previous reply reached the output limit. None of its tools ran. '
                    'Continue using current workspace and simulator status; do not replay any action.'})
                continue
            messages.append(message)
            calls = message.get('tool_calls') or []
            if not calls:
                if choice.get('finish_reason') != 'stop':
                    raise RuntimeError('Model ended without tools: ' + str(choice.get('finish_reason')))
                print(message.get('content') or 'Agent completed', flush=True)
                return 0
            if len(calls) != 1:
                if parallel_tool_rejections >= 1:
                    raise RuntimeError(
                        'Model repeated parallel tool calls after the single-call correction; '
                        'no parallel calls were executed'
                    )
                parallel_tool_rejections += 1
                messages.extend(parallel_tool_rejection_messages(calls))
                log('parallel_tool_calls_rejected', {
                    'call_count': len(calls), 'executed_count': 0,
                    'correction_turn': parallel_tool_rejections,
                })
                continue
            call = calls[0]
            fn = call['function']
            if deadline is not None:
                remaining(deadline)
            log('tool_call', {'id': call['id'], 'name': fn['name'], 'arguments': fn['arguments']})
            try:
                output, images = execute_tool(root, args.container, args.container_socket,
                                              fn['name'], json.loads(fn['arguments']))
            except subprocess.TimeoutExpired:
                output, images = {'error': 'Command timed out; it may have run. Inspect status; do not repeat.'}, []
            except (ValueError, KeyError, OSError) as error:
                output, images = {'error': str(error)}, []
            content = json.dumps(output, ensure_ascii=False).replace(key, '[redacted]')
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': content})
            log('tool_result', {'id': call['id'], 'name': fn['name'], 'content': content})
            if images:
                if len(images) > 2:
                    raise RuntimeError('Too many images in one response')
                messages.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Requested native camera images, unchanged.'}, *images]})
    finally:
        (log_dir / 'usage.json').write_text(json.dumps(
            {'provider': 'openrouter', 'model': args.model, 'receipts': receipts, **totals},
            indent=2) + '\n', encoding='utf-8')
    raise RuntimeError('OpenRouter agent inference-round ceiling reached')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--effort', default='high')
    parser.add_argument('--phase', choices=('playground', 'test'), required=True)
    parser.add_argument('--role', choices=('solve', 'repair'), required=True)
    parser.add_argument('--handoff')
    parser.add_argument('--container', required=True)
    parser.add_argument('--container-socket', default='/tmp/session.sock')
    parser.add_argument('--image', default='embodiedevo-agent:latest')
    parser.add_argument('--max-rounds', type=int, default=600)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except BaseException as error:
        print(f'{type(error).__name__}: {str(error)[:1000]}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
