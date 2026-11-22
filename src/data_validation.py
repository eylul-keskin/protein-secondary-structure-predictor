from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import pandas as pd


@dataclass
class SplitValidationSummary:
    """Captures the main quality checks for one split."""

    split_name: str
    file_path: str
    total_rows: int
    kept_rows: int
    removed_rows: int
    missing_sequences: int
    missing_targets: int
    empty_sequences: int
    empty_targets: int
    length_mismatches: int
    invalid_label_rows: int
    duplicate_rows: int
    duplicate_sequences: int
    sequence_length_stats: dict[str, float]
    unique_sequence_characters: list[str]
    unique_target_labels: list[str]
    label_distribution: dict[str, int]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _sequence_length_stats(lengths: pd.Series) -> dict[str, float]:
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(lengths.median()),
    }


def validate_and_clean_split(
    dataframe: pd.DataFrame,
    split_name: str,
    file_path: Path,
    sequence_column: str,
    target_column: str,
    allowed_label_characters: set[str] | None = None,
) -> tuple[pd.DataFrame, SplitValidationSummary]:
    """Validates a split and removes rows that are unusable for training/evaluation."""

    working = dataframe.copy()
    working[sequence_column] = working[sequence_column].fillna("").astype(str)
    working[target_column] = working[target_column].fillna("").astype(str)

    missing_sequences = int(dataframe[sequence_column].isna().sum())
    missing_targets = int(dataframe[target_column].isna().sum())
    empty_sequences = int((working[sequence_column].str.len() == 0).sum())
    empty_targets = int((working[target_column].str.len() == 0).sum())
    length_mismatches = int((working[sequence_column].str.len() != working[target_column].str.len()).sum())

    if allowed_label_characters is None:
        invalid_label_mask = pd.Series(False, index=working.index)
    else:
        invalid_label_mask = working[target_column].apply(
            lambda label_string: any(label not in allowed_label_characters for label in label_string)
        )
    invalid_label_rows = int(invalid_label_mask.sum())

    unusable_rows = (
        (working[sequence_column].str.len() == 0)
        | (working[target_column].str.len() == 0)
        | (working[sequence_column].str.len() != working[target_column].str.len())
        | invalid_label_mask
    )

    cleaned = working.loc[~unusable_rows].reset_index(drop=True)
    label_distribution = Counter("".join(cleaned[target_column].tolist()))
    lengths = cleaned[sequence_column].str.len()
    notes: list[str] = []

    duplicate_rows = int(cleaned.duplicated().sum())
    duplicate_sequences = int(cleaned.duplicated(subset=[sequence_column]).sum())
    if duplicate_rows:
        notes.append(
            f"{duplicate_rows} duplicate rows were detected and kept because this split is treated as a provided benchmark."
        )
    if duplicate_sequences and duplicate_sequences != duplicate_rows:
        notes.append(f"{duplicate_sequences} duplicate sequences were detected in the cleaned split.")
    if invalid_label_rows:
        notes.append(f"{invalid_label_rows} rows were removed because they contained labels outside the SST8 mapping.")
    if int(unusable_rows.sum()):
        notes.append(
            f"{int(unusable_rows.sum())} rows were removed because of missing values, empty fields, or length mismatches."
        )
    if not notes:
        notes.append("No unusable rows were removed from this split.")

    summary = SplitValidationSummary(
        split_name=split_name,
        file_path=str(file_path),
        total_rows=int(len(dataframe)),
        kept_rows=int(len(cleaned)),
        removed_rows=int(unusable_rows.sum()),
        missing_sequences=missing_sequences,
        missing_targets=missing_targets,
        empty_sequences=empty_sequences,
        empty_targets=empty_targets,
        length_mismatches=length_mismatches,
        invalid_label_rows=invalid_label_rows,
        duplicate_rows=duplicate_rows,
        duplicate_sequences=duplicate_sequences,
        sequence_length_stats=_sequence_length_stats(lengths),
        unique_sequence_characters=sorted(set("".join(cleaned[sequence_column].tolist()))),
        unique_target_labels=sorted(label_distribution.keys()),
        label_distribution=dict(sorted(label_distribution.items())),
        notes=notes,
    )
    return cleaned, summary


def compute_split_overlaps(splits: dict[str, pd.DataFrame], sequence_column: str) -> list[dict[str, int | str]]:
    """Counts exact sequence overlaps across pairs of splits."""

    overlaps: list[dict[str, int | str]] = []
    for split_a, split_b in combinations(splits.keys(), 2):
        overlap_count = len(set(splits[split_a][sequence_column]).intersection(set(splits[split_b][sequence_column])))
        overlaps.append({"split_a": split_a, "split_b": split_b, "overlap_count": overlap_count})
    return overlaps
