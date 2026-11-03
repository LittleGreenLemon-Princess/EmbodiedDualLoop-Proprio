"""Load and seal a complete parameterized native-task program, not scene state."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from orchestrator.dual_loop.model import TaskBrief

MANIFEST = 'skills/task_program.json'
PROFILE = 'dual_loop_frozen_task'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_program(workspace, native_task):
    workspace = Path(workspace).resolve()

    def read(name, suffix):
        if not isinstance(name, str):
            raise ValueError('Program paths must be strings')
        relative = Path(name)
        path = workspace / relative
        if (not relative.parts or relative.is_absolute() or '..' in relative.parts
                or relative.parts[0] != 'skills' or relative.suffix != suffix
                or path.resolve() != path or not path.is_file()):
            raise ValueError('Frozen program must use real files inside skills/')
        return path.read_text()

    raw = json.loads(read(MANIFEST, '.json'))
    if raw.get('schema') != 'frozen-task-v1' or raw.get('native_task') != native_task:
        raise ValueError('Frozen task schema or native task mismatch')
    task = dict(raw['task'])
    entry, includes = task.pop('entry'), task.pop('includes', [])
    if not isinstance(includes, list):
        raise ValueError('includes must be a list')
    names = [*includes, entry]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate program sources')
    task['sources'] = [{'path': n, 'code': read(n, '.py')} for n in names]
    brief = TaskBrief.from_request(task)
    selector = dict(raw.get('selector', {'kind': 'jev', 'timeout': 8.}))
    if selector.get('kind') not in ('jev', 'jev_factorized', 'openrouter'):
        raise ValueError('Frozen execution requires a model-backed System 1 selector')
    from orchestrator.dual_loop.policies import make_policy
    make_policy(selector)  # Validate settings; does not call the provider.
    observation_profile = raw.get('observation_profile', 'baseline')
    if observation_profile not in ('baseline', 'proprioception-v1'):
        raise ValueError('Unknown frozen observation profile')
    if brief.min_confidence < .5:
        raise ValueError('Frozen execution requires min_confidence >= .5')
    interventions = raw.get('max_system2_interventions', 2)
    if type(interventions) is not int or not 0 <= interventions <= 3:
        raise ValueError('max_system2_interventions must be 0..3')
    identity = {'manifest': raw, 'sources': [
        {'path': n, 'sha256': hashlib.sha256(s['code'].encode()).hexdigest()}
        for n, s in zip(names, brief.sources)]}
    return {'brief': brief, 'selector': selector, 'max_interventions': interventions,
            'observation_profile': observation_profile,
            'sha256': digest(identity), 'identity': identity}


def execution_signature(brief):
    value = asdict(brief)
    value.pop('task_id'); value.pop('revision')
    value['sources'] = [hashlib.sha256(s['code'].encode()).hexdigest() for s in brief.sources]
    return digest(value)


def validate_development_export(workspace, native_task):
    """Only seal code/parameters that really ran through the development runtime."""
    program = load_program(workspace, native_task)
    expected = execution_signature(program['brief'])
    for path in sorted(Path(workspace).glob('observations/*/system1/task.json'), reverse=True):
        tested = TaskBrief.from_request(json.loads(path.read_text()))
        report_path = path.parent / 'report.json'
        if execution_signature(tested) == expected and report_path.exists():
            report = json.loads(report_path.read_text())
            # Validation establishes execution/provenance, not task success.
            if report.get('reason') in ('program_error', 'runtime_error'):
                continue
            context_seal = {}
            if program['selector']['kind'] == 'openrouter':
                context_path = path.parent.parent / 'system1-context.json'
                if (report.get('reason') == 'selector_error' or report.get('action_calls', 0) < 1
                        or not context_path.is_file()):
                    continue
                context = json.loads(context_path.read_text(encoding='utf-8'))
                if context != {'selector': program['selector'],
                               'observation_profile': program['observation_profile']}:
                    continue
                context_seal['development_context_sha256'] = hashlib.sha256(context_path.read_bytes()).hexdigest()
            return {k: v for k, v in {
                'schema': 'frozen-task-seal-v1', 'native_task': native_task,
                'program_sha256': program['sha256'], 'manifest': MANIFEST,
                'development_report': str(report_path.relative_to(workspace)),
                'development_status': report.get('status'),
                'development_reason': report.get('reason'),
                'subtask_verified': report.get('subtask_verified', False),
                **context_seal,
            }.items()}
    raise ValueError('Exported full program was not executed with the same code and parameters '
                     'during Playground; preserve the run and do not start Test')
