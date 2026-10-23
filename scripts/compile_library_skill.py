"""Compile a pinned library bundle to a normal delegate task (no execution)."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from orchestrator.dual_loop.skill_library import SkillLibrary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--skill', default='thin-rim-pick')
    parser.add_argument('--version', default='1.0.0')
    parser.add_argument('--embodiment', default='robodojo-arx-x5')
    parser.add_argument('--signals', nargs='+', required=True)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    library = SkillLibrary(args.library_root)
    brief = library.compile(args.skill, args.version, embodiment=args.embodiment,
                            available_signals=args.signals,
                            parameters=json.loads(args.parameters.read_text(encoding='utf-8')),
                            task_id=args.task_id, goal=args.goal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite a submitted or frozen task.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(asdict(brief), stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
