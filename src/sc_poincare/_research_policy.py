"""Execution policy for maintained research CLIs, not a training objective.

Closed inputs remain readable by audit/evaluation tools. This is a fail-closed
CLI convention, not an OS sandbox or protection against arbitrary Python code.
Core array-based fitting does not import or infer filesystem provenance here.
"""
from __future__ import annotations

from pathlib import Path

CLOSED_RESEARCH_MARKER = 'CLOSED_RESEARCH.json'


def ensure_active_paths(*paths):
    """Reject inputs/resume/jobs/outputs inside a marked closed research tree.

    Check both lexical and resolved ancestry so a symlink cannot silently
    bypass the convention. Nonexistent output children are checked as well.
    No files are opened, copied, deleted or changed by this check.
    """
    for value in paths:
        if value is None:
            continue
        path = Path(value).expanduser().absolute()
        for candidate in (path, path.resolve()):
            for parent in (candidate, *candidate.parents):
                marker = parent / CLOSED_RESEARCH_MARKER
                if marker.is_file():
                    raise ValueError(
                        f'Closed research tree is excluded from training/replay: {path}. '
                        f'Closure marker: {marker}. Read-only audits/evaluation remain allowed; '
                        'a new experiment requires a separately authorized protocol and active inputs.'
                    )
