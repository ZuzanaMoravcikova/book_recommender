
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    cleaned_books: Path
    cleaned_ratings: Path
    goodreads_books: Path


def _first_existing(root: Path, candidates: Iterable[str]) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return root / list(candidates)[0]


def resolve_project_root(start: Path | None = None) -> Path:
    env_root = os.getenv('BOOK_APP_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend([
        Path.cwd().resolve(),
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    ])

    for candidate in candidates:
        if (candidate / 'data').exists() and (candidate / 'methods_notebooks').exists():
            return candidate
    return candidates[0]


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    root = resolve_project_root(root)
    return ProjectPaths(
        root=root,
        cleaned_books=_first_existing(root, [
            'data/cleaned/cleaned_books.csv',
            'data/cleaned_books.csv',
        ]),
        cleaned_ratings=_first_existing(root, [
            'data/cleaned/cleaned_ratings.csv',
            'data/cleaned_ratings.csv',
        ]),
        goodreads_books=_first_existing(root, [
            'data_B/dataset_goodreads_filtered_description.csv',
            'data_B/goodreads_books.csv',
        ]),
    )
