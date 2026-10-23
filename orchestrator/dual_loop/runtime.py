"""System 1 owns repeated observe/select/execute; System 2 receives the report."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path

from orchestrator.files import append_jsonl, write_json

from .evidence import save_frames


class ExecutorLoop:
    def __init__(self, brief, backend, policy, run_dir, *, verifier=None, clock=time.monotonic,
                 state_transform=None):
        self.brief, self.backend, self.policy = brief, backend, policy
        self.run_dir = Path(run_dir)
        self.clock, self.verifier = clock, verifier
        self.state_transform = state_transform
        self.decisions = self.actions = 0
        self.events = []
        self.frames = {}
        self.latest = None

    def _record_execution(self, kind, result, calls):
        # Physical calls are spent even if saving their feedback subsequently fails.
        self.actions += calls
        index = len(self.events)
        folder = self.run_dir / f'{index:06d}_{kind}'
        folder.mkdir(parents=True)
        (folder / 'code.py').write_text(result.code)
        (folder / 'stdout.txt').write_text(result.stdout)
        (folder / 'stderr.txt').write_text(result.stderr)
        saved = save_frames(result.frames, folder / 'frames')
        for camera, paths in saved.items():
            self.frames.setdefault(camera, []).extend(paths)
        return {'execution_ref': str(folder.relative_to(self.run_dir)),
                'code_sha256': hashlib.sha256(result.code.encode()).hexdigest(),
                'code_ok': result.ok, 'action_calls': calls,
                'frame_refs': {camera: [str(path.relative_to(self.run_dir)) for path in paths]
                               for camera, paths in saved.items()},
                'terminated': result.terminated, 'truncated': result.truncated,
                'native_success': result.success}

    def _record(self, event):
        event = {'seq': len(self.events), 'elapsed_seconds': self.clock() - self.started, **event}
        append_jsonl(self.run_dir / 'events.jsonl', event)
        self.events.append(event)

    def _finish(self, status, reason):
        report = {
            'protocol': 'dual-loop-v1', 'task_id': self.brief.task_id,
            'experiment_condition': getattr(self.backend, 'experiment_condition', 'dual-loop-v1'),
            'revision': self.brief.revision, 'goal': self.brief.goal,
            'status': status, 'reason': reason, 'decisions': self.decisions,
            'action_calls': self.actions, 'elapsed_seconds': self.clock() - self.started,
            'subtask_verified': status == 'verified_subtask',
            'latest': self.latest, 'events_ref': 'events.jsonl',
            'sources': [{'path': item['path'], 'sha256': hashlib.sha256(
                item['code'].encode()).hexdigest()} for item in self.brief.sources],
            'evidence': self._evidence(),
        }
        write_json(self.run_dir / 'report.json', report)
        return report

    def _evidence(self):
        # Retain all events on disk; select first, recent, and failure records for System 2.
        chosen = {0, *range(max(0, len(self.events) - 4), len(self.events))}
        chosen.update(i for i, e in enumerate(self.events) if e.get('code_ok') is False)
        return [self.events[i] for i in sorted(chosen) if i < len(self.events)]

    def _terminal(self, snapshot, result):
        if result.terminated or result.truncated:
            return ('native_task_succeeded' if result.success else 'needs_system2',
                    'native_success' if result.success else 'native_episode_ended')
        if not result.ok or snapshot is None:
            return 'needs_system2', 'program_error'
        status = snapshot['context'].get('status', 'active')
        if status in ('needs_verification', 'completed', 'done'):
            return self._verify(snapshot)
        if status != 'active':
            return 'needs_system2', snapshot['context'].get('reason') or status
        return None

    def _verify(self, snapshot):
        # Model assertions and motor convergence alone cannot establish physical success.
        if self.verifier is not None and self.verifier(snapshot) is True:
            return 'verified_subtask', 'physical_evidence_verified'
        return 'awaiting_verification', 'physical_evidence_required'

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.started = self.clock()
        write_json(self.run_dir / 'task.json', asdict(self.brief))
        try:
            snapshot, result, calls = self.backend.initialize()
            self.latest = snapshot
            self._record({'kind': 'initialize', **self._record_execution('initialize', result, calls)})
            terminal = self._terminal(snapshot, result)
            if terminal:
                return self._finish(*terminal)
            while True:
                if self.decisions >= self.brief.max_decisions:
                    return self._finish('needs_system2', 'decision_budget')
                if self.actions >= self.brief.max_action_calls:
                    return self._finish('needs_system2', 'action_call_budget')
                if self.clock() - self.started >= self.brief.max_seconds:
                    return self._finish('needs_system2', 'time_budget')
                snapshot, result, calls = self.backend.refresh()
                observed_at = self.clock()
                self.latest = snapshot
                self._record({'kind': 'observe', **self._record_execution('observe', result, calls)})
                terminal = self._terminal(snapshot, result)
                if terminal:
                    return self._finish(*terminal)
                options = snapshot['options']
                if not options:
                    return self._finish('needs_system2', 'no_available_actions')
                state = {'goal': self.brief.goal, 'current': snapshot['context'],
                         'recent_effects': self.events[-4:]}
                self.decisions += 1
                try:
                    policy_state = self.state_transform(state) if self.state_transform else state
                    if getattr(self.backend, 'proprioception', None):
                        policy_state = {**policy_state,
                                        'proprioception': snapshot['context']['proprioception']}
                    decision = self.policy.select(policy_state, options)
                    decision.validate(options)
                except Exception as error:
                    self._record({'kind': 'selector_error', 'error_type': type(error).__name__})
                    return self._finish('needs_system2', 'selector_error')
                self._record({'kind': 'decision', 'decision': asdict(decision),
                              'latency_seconds': self.clock() - observed_at})
                if self.clock() - self.started >= self.brief.max_seconds:
                    return self._finish('needs_system2', 'time_budget')
                if self.clock() - observed_at > self.brief.max_state_age_seconds:
                    return self._finish('needs_system2', 'stale_observation')
                if decision.confidence < self.brief.min_confidence:
                    return self._finish('needs_system2', 'low_confidence')
                if decision.judgement == 'done':
                    return self._finish(*self._verify(snapshot))
                if decision.judgement in ('stuck', 'unknown') or decision.action == 'escalate':
                    reason = ('policy_requested' if decision.action == 'escalate'
                              else 'policy_' + decision.judgement)
                    return self._finish('needs_system2', reason)
                snapshot, result, calls = self.backend.tick(decision.action)
                self.latest = snapshot
                self._record({'kind': 'action', 'action': decision.action,
                              **self._record_execution('tick', result, calls),
                              'effect': snapshot['feedback'] if snapshot else None})
                terminal = self._terminal(snapshot, result)
                if terminal:
                    return self._finish(*terminal)
        except Exception as error:
            self._record({'kind': 'runtime_error', 'error_type': type(error).__name__})
            return self._finish('needs_system2', 'runtime_error')
