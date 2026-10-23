"""Versioned, hash-locked skill bundles compiled into ordinary TaskBrief inputs.

Concepts from ASPIRE's reusable knowledge: scope, requirements, provenance and
locked revisions. No host exec, automatic learning, or cross-Test persistence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .model import TaskBrief


class SkillLibrary:
    def __init__(self, root, manifest='integrations/skills/library.json'):
        self.root = Path(root).resolve()
        self.manifest = json.loads(self._path(manifest).read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'dual-loop-skill-library-v1':
            raise ValueError('Unknown library schema')
        self.skills = self.manifest['skills']
        ids = [(s['id'], s['version']) for s in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate skill revision')

    def _path(self, relative):
        if Path(relative).is_absolute():
            raise ValueError('Source paths must be relative')
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Source escapes library root')
        return path

    def search(self, *, embodiment, capabilities=()):
        """Deterministic metadata retrieval; does not assert physical applicability."""
        required = set(capabilities)
        return [s for s in self.skills if embodiment in s['embodiments']
                and required <= set(s['capabilities']) and s['status'] == 'supplied']

    def compile(self, skill_id, version, *, embodiment, available_signals, parameters,
                task_id, goal):
        skill = next((s for s in self.skills if (s['id'], s['version']) == (skill_id, version)), None)
        if skill is None:
            raise ValueError('Unknown pinned skill revision')
        if skill['status'] != 'supplied' or embodiment not in skill['embodiments']:
            raise ValueError('Skill is unavailable for this embodiment')
        missing = set(skill['required_signals']) - set(available_signals)
        if missing:
            raise ValueError('Missing public signals: ' + ', '.join(sorted(missing)))
        sources = []
        for item in skill['sources']:
            raw = self._path(item['path']).read_text(encoding='utf-8').encode('utf-8')
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Skill source hash mismatch: ' + item['path'])
            sources.append({'path': item['path'], 'code': raw.decode('utf-8')})
        return TaskBrief(task_id, goal, tuple(sources), parameters=parameters)
