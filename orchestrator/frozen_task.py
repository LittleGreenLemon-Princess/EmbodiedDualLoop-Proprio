"""Agent-free normal execution; start real System 2 only after an evidenced exit."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from orchestrator.config import config_from_dict
from orchestrator.dual_loop.model import TaskBrief
from orchestrator.files import append_jsonl, hashes, write_json
from orchestrator.frozen_program import digest, load_program
from orchestrator.session_client import request, source_snapshot


def knowledge_hashes(root):
    # Validate links first, including links inside caches. Ignore only generated
    # bytecode, never Python sources or arbitrary files hidden in __pycache__.
    return {name: value for name, value in hashes(root).items()
            if not ('__pycache__' in Path(name).parts[:-1]
                    and Path(name).suffix == '.pyc')}


class RepairValidationError(ValueError):
    """Safe schema diagnostics, without proposal text, code or host paths."""
    def __init__(self, field, message):
        super().__init__(message)
        self.field = field


def load_repair(workspace, proposal, base, revision, *, base_selector, current=None):
    try:
        raw = json.loads(Path(proposal).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise RepairValidationError('proposal', 'Read a valid UTF-8 JSON proposal') from error
    if not isinstance(raw, dict):
        raise RepairValidationError('proposal', 'Repair proposal must be a JSON object')
    if raw.get('stop') is True:
        return None
    if not isinstance(raw.get('task'), dict):
        raise RepairValidationError('task', 'Repair requires a task object; diagnostics alone are not executable')
    changes = dict(raw['task'])
    entry = changes.pop('entry', None)
    if not isinstance(entry, str) or not entry.strip():
        raise RepairValidationError('task.entry', 'task.entry must name a workspace Python source')
    # Keep omitted fields at the current revision, rather than TaskBrief defaults
    # that can enlarge budgets or erase parameters/dependencies.
    task = asdict(current if current is not None else base)
    includes = changes.pop('includes', [s['path'] for s in task['sources'][:-1]])
    if (not isinstance(includes, list)
            or any(not isinstance(name, str) or not name.strip() for name in includes)):
        raise RepairValidationError('task.includes', 'task.includes must be a list of Python source paths')
    if len(set([*includes, entry])) != len(includes) + 1:
        raise RepairValidationError('task.includes', 'Repair sources must not contain duplicate paths')
    if 'sources' in changes:
        raise RepairValidationError('task.sources', 'Use task.entry and task.includes, not embedded sources')
    task.update(changes)
    try:
        task['sources'] = source_snapshot(workspace, entry, includes)
    except (OSError, UnicodeError, ValueError, TypeError, IndexError) as error:
        raise RepairValidationError('task.sources', 'Repair sources must be readable UTF-8 .py files inside submission/ or skills/') from error
    task.update(task_id=base.task_id, goal=base.goal, revision=revision)
    try:
        repaired = TaskBrief.from_request(task)
    except (ValueError, TypeError, SyntaxError, OverflowError) as error:
        raise RepairValidationError('task', 'Repair task must have valid fields, finite limits, object parameters and Python skill_init/context/options/tick entry points') from error
    for key in ['max_decisions', 'max_action_calls', 'max_seconds', 'max_state_age_seconds']:
        if getattr(repaired, key) > getattr(base, key):
            raise RepairValidationError('task.' + key, 'Repair cannot enlarge ' + key)
    if repaired.min_confidence < base.min_confidence:
        raise RepairValidationError('task.min_confidence', 'Repair cannot lower the confidence threshold')
    selector = raw.get('selector', base_selector)
    if not isinstance(selector, dict) or selector != base_selector:
        raise RepairValidationError('selector', 'Repair must inherit the complete frozen selector configuration unchanged')
    return repaired, dict(base_selector)


def drive(workspace, native_task, expected_sha256, call, repair, output, *, controller_key):
    """All callbacks share ONE simulator; no reset or cross-scene code promotion."""
    workspace, output = Path(workspace), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    program = load_program(workspace, native_task)
    if program['sha256'] != expected_sha256:
        raise ValueError('Frozen task hash differs from the development seal')
    knowledge = {part: knowledge_hashes(workspace / part) for part in ['skills', 'lessons']}
    base = current = program['brief']
    selector = program['selector']
    summary = {'condition': 'fixed-base-with-local-system2-repair-v1',
               'initial_program_sha256': expected_sha256, 'native_task': native_task,
               'system2_interventions': 0, 'delegations': 0, 'native_success': False,
               'primary_success': False, 'repairs': [], 'repair_agent_seconds': 0.}

    def log(kind, **fields):
        append_jsonl(output / 'events.jsonl', {'kind': kind,
                     'elapsed_seconds': time.monotonic() - started, **fields})
        summary['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'result.json', summary)

    def unchanged():
        if (load_program(workspace, native_task)['sha256'] != expected_sha256
                or knowledge != {part: knowledge_hashes(workspace / part) for part in knowledge}):
            raise ValueError('Frozen base knowledge changed within a Test scene')

    def rpc(payload):
        return call({**payload, 'controller_key': controller_key})

    while True:
        unchanged()
        payload = {'op': 'delegate', 'task': asdict(current), 'selector': selector}
        log('dispatch', revision=current.revision, program_sha256=digest(payload))
        response = rpc(payload)
        summary['delegations'] += 1
        if 'system1' not in response:
            summary.update(reason='session_rejected_request', status='failed')
            log('rejected', error=response.get('error'))
            # An error receipt is not evidence that a physical operation can be retried.
            raise RuntimeError('Session rejected fixed-task request; no automatic retry')
        report = response['system1']
        summary['native_instruction'] = report['goal']
        summary['last_report'] = response.get('report')
        summary['native_success'] = bool(response.get('success'))
        if summary['delegations'] == 1:
            summary['primary_success'] = summary['native_success']
            summary['primary_exit_reason'] = report['reason']
        log('system1_return', revision=current.revision, status=report['status'],
            reason=report['reason'], report=response.get('report'),
            native_success=summary['native_success'])
        if summary['native_success']:
            summary.update(status='finished', reason='native_success')
            break
        if response.get('terminated') or response.get('truncated') or response.get('closed'):
            summary.update(status='finished', reason='native_episode_ended')
            break
        if summary['system2_interventions'] >= program['max_interventions']:
            summary.update(status='finished', reason='system2_intervention_limit')
            break
        status = rpc({'op': 'status'})
        if (status.get('executions_remaining', 0) <= 0
                or status.get('native_steps_remaining', 0) <= 0):
            summary.update(status='finished', reason='session_budget_exhausted')
            break
        index = summary['system2_interventions'] + 1
        template = asdict(current)
        sources = template.pop('sources')
        template.update(entry=sources[-1]['path'], includes=[s['path'] for s in sources[:-1]],
                        goal=report['goal'], revision=base.revision + index)
        handoff = {'native_task': native_task, 'initial_program_sha256': expected_sha256,
                   'intervention': index, 'trigger': report['reason'],
                   'report': response.get('report'), 'system1': report, 'status': status,
                   'base_limits': {k: getattr(base, k) for k in ['max_decisions',
                       'max_action_calls', 'max_seconds', 'max_state_age_seconds', 'min_confidence']},
                   'proposal_path': f'submission/repair_{index:04d}.json',
                   'proposal_template': {'task': template},
                   'proposal_contract': {
                       'required': ['task'], 'task_required': ['entry'],
                       'omitted_task_fields': 'Inherit the current revision, including parameters, includes and limits.',
                       'selector_rule': 'Omit selector to inherit the complete frozen configuration; any change is forbidden.',
                       'source_rule': 'Template paths name the current sources. Write changed code under submission/ and update entry/includes; never edit skills/.',
                       'stop_example': {'stop': True, 'reason': 'No evidenced repair is available.'}},
                   'rule': 'Same physical scene; propose one local repair, no movement, '
                           'no reset/delegate/finish, no frozen-library changes.'}
        handoff_path = workspace / 'scratch' / f'handoff_{index:04d}.json'
        write_json(handoff_path, handoff)
        summary['system2_interventions'] = index
        log('system2_start', intervention=index, trigger=report['reason'])
        repair_started = time.monotonic()
        try:
            proposal = repair(handoff_path, handoff)
        finally:
            summary['repair_agent_seconds'] += time.monotonic() - repair_started
            log('system2_return', intervention=index)
        unchanged()
        if proposal is None:
            summary.update(status='finished', reason='system2_could_not_repair')
            break
        try:
            revised = load_repair(workspace, proposal, base, base.revision + index,
                                  base_selector=program['selector'], current=current)
        except RepairValidationError as error:
            summary.update(status='finished', reason='invalid_repair_proposal')
            log('invalid_repair', error_type=type(error).__name__, field=error.field,
                message=str(error), cause_type=type(error.__cause__).__name__ if error.__cause__ else None)
            break
        if revised is None:
            summary.update(status='finished', reason='system2_reported_unresolved_failure')
            break
        next_brief, next_selector = revised
        material = lambda b: digest({'sources': [s['code'] for s in b.sources],
                                    'parameters': b.parameters})
        if material(next_brief) == material(current):
            summary.update(status='finished', reason='unchanged_repair_rejected')
            break
        summary['repairs'].append({'intervention': index, 'trigger': report['reason'],
                                  'proposal': str(Path(proposal).relative_to(workspace)),
                                  'revision': next_brief.revision})
        current, selector = next_brief, next_selector
        log('repair_accepted', intervention=index, revision=current.revision)
    unchanged()
    rpc({'op': 'finish'})
    log('terminal', reason=summary['reason'], native_success=summary['native_success'])
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--home', type=Path, required=True)
    p.add_argument('--controller-key-file', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    args = p.parse_args()
    cfg = config_from_dict(json.loads(args.config.read_text()))
    workspace = args.episode / 'workspace'
    output = args.episode / 'fixed-task'

    def repair(path, handoff):
        index = handoff['intervention']
        logs = args.episode / 'repairs' / f'{index:04d}' / 'agent'
        command = [sys.executable, '-m', 'orchestrator.local_agent',
                   '--episode', str(args.episode), '--socket', str(args.socket),
                   '--home', str(args.home / f'repair_{index:04d}'),
                   '--log-dir', str(logs), '--model', cfg.agent.model,
                   '--phase', 'test', '--effort', cfg.agent.reasoning_effort,
                   '--role', 'repair', '--handoff', str(path.relative_to(workspace))]
        # Inherit the controller's process group: supervisor cleanup kills the
        # repair CLI and its children too. The overall scene watchdog still applies.
        rc = subprocess.run(command, check=False).returncode
        proposal = workspace / handoff['proposal_path']
        return proposal if rc == 0 and proposal.is_file() else None

    try:
        drive(workspace, cfg.simulator.env_task, args.expected_sha256,
              lambda payload: request(payload, socket_path=str(args.socket)), repair, output,
              controller_key=args.controller_key_file.read_text().strip())
        return 0
    except Exception as error:
        write_json(output / 'error.json', {'error_type': type(error).__name__,
                                         'message': str(error)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
