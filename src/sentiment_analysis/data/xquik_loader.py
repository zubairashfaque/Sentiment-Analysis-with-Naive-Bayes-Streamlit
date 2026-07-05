"""
Utilities for loading Xquik CSV exports for batch sentiment prediction.
"""

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PathLike = Union[str, Path]

TEXT_COLUMN_ALIASES = {
    "text",
    "tweet text",
    "tweet_text",
    "tweettext",
    "comment",
    "comments",
    "review",
    "reviews",
    "feedback",
    "message",
    "body",
}


def normalize_column_name(column: object) -> str:
    """Normalize CSV header names for alias matching."""
    return " ".join(str(column).replace("_", " ").strip().lower().split())


def find_text_column(columns: Iterable[object]) -> str:
    """
    Find the text-bearing column in a generic or Xquik CSV export.

    Exact aliases are preferred over broader substring matches so metadata
    columns such as "tweet_created_at" are not selected before the text column.
    """
    column_names = list(columns)
    normalized = {normalize_column_name(column): str(column) for column in column_names}

    for alias in TEXT_COLUMN_ALIASES:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized:
            return normalized[normalized_alias]

    for column in column_names:
        normalized_column = normalize_column_name(column)
        if any(token in normalized_column.split() for token in ("text", "tweet", "comment", "review", "feedback")):
            return str(column)

    raise ValueError("CSV must contain a text, tweet text, comment, review, or feedback column")


def load_xquik_texts(source: Union[PathLike, pd.DataFrame]) -> list[str]:
    """
    Load non-empty texts from an Xquik or generic CSV export.

    Args:
        source: A CSV path or an already loaded DataFrame.

    Returns:
        Clean text values ready for batch prediction.
    """
    frame = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    text_column = find_text_column(frame.columns)
    texts = frame[text_column].fillna("").astype(str).str.strip()
    texts = [text for text in texts.tolist() if text]
    if not texts:
        raise ValueError("CSV text column does not contain any non-empty values")
    return texts


def prepare_batch_prediction_frame(source: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    """Return a normalized DataFrame with the project's expected text column."""
    return pd.DataFrame({"text": load_xquik_texts(source)})
