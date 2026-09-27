"""Validation and normalization for five-option question data."""

from collections.abc import Iterable

import pandas as pd

OPTIONS = ("A", "B", "C", "D", "E")
LABEL_TO_INDEX = {label: index for index, label in enumerate(OPTIONS)}


def validate_questions(frame: pd.DataFrame, require_answer: bool = False) -> pd.DataFrame:
    """Return a normalized copy and fail early when the input contract is invalid."""
    required = {"prompt", *OPTIONS}
    if require_answer:
        required.add("answer")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = frame.copy()
    normalized["prompt"] = normalized["prompt"].fillna("").astype(str)
    for option in OPTIONS:
        normalized[option] = normalized[option].fillna("").astype(str)
    if require_answer:
        normalized["answer"] = normalized["answer"].map(normalize_label)
        normalized["hard_label"] = normalized["answer"].map(LABEL_TO_INDEX)
    return normalized


def normalize_label(value: object) -> str:
    """Normalize an answer label and reject values outside A-E."""
    label = str(value).strip().upper()
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"Invalid answer label: {value!r}")
    return label


def build_queries(frame: pd.DataFrame, repeat_prompt: int = 1) -> list[str]:
    """Build deterministic retrieval queries from prompts and all answer options."""
    normalized = validate_questions(frame)
    queries: Iterable[str] = (
        (" ".join([row.prompt] * repeat_prompt + [getattr(row, option) for option in OPTIONS])).strip()
        for row in normalized.itertuples(index=False)
    )
    return list(queries)