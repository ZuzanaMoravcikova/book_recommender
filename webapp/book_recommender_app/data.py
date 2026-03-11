
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import ProjectPaths, get_project_paths


def load_books(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.cleaned_books)


def load_ratings(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.cleaned_ratings)


def load_goodreads(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.goodreads_books)


def summarize_datasets(paths: ProjectPaths) -> Dict[str, object]:
    summary: Dict[str, object] = {
        'root': str(paths.root),
        'cleaned_books_exists': paths.cleaned_books.exists(),
        'cleaned_ratings_exists': paths.cleaned_ratings.exists(),
        'goodreads_books_exists': paths.goodreads_books.exists(),
        'cleaned_books_path': str(paths.cleaned_books),
        'cleaned_ratings_path': str(paths.cleaned_ratings),
        'goodreads_books_path': str(paths.goodreads_books),
    }

    if paths.cleaned_books.exists():
        books = load_books(paths)
        summary['cleaned_books_shape'] = books.shape
        summary['cleaned_books_columns'] = list(books.columns)

    if paths.cleaned_ratings.exists():
        ratings = load_ratings(paths)
        summary['cleaned_ratings_shape'] = ratings.shape
        summary['cleaned_ratings_columns'] = list(ratings.columns)

    if paths.goodreads_books.exists():
        goodreads = load_goodreads(paths)
        summary['goodreads_books_shape'] = goodreads.shape
        summary['goodreads_books_columns'] = list(goodreads.columns)

    return summary
