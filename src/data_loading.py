from __future__ import annotations

from typing import Iterable

import pandas as pd


SEQUENCE_COLUMN_CANDIDATES = ("seq", "sequence", "amino_acid_sequence")
SST8_COLUMN_CANDIDATES = ("sst8", "ss8", "secondary_structure_8", "secondary_structure_sst8")
SST3_COLUMN_CANDIDATES = ("sst3", "ss3", "secondary_structure_3", "secondary_structure_sst3")


def load_csv(path: str | pd.io.common.FilePath) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""

    return pd.read_csv(path)


def _normalize_columns(columns: Iterable[str]) -> dict[str, str]:
    return {column.lower().strip(): column for column in columns}


def discover_sequence_column(columns: Iterable[str]) -> str:
    """Finds the amino-acid sequence column using simple name heuristics."""

    normalized = _normalize_columns(columns)
    for candidate in SEQUENCE_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    for normalized_name, original_name in normalized.items():
        if "seq" in normalized_name:
            return original_name
    raise ValueError("Could not discover the amino-acid sequence column.")


def discover_target_column(columns: Iterable[str], preferred: str = "sst8") -> str:
    """Finds the requested target column, preferring SST8 over SST3."""

    normalized = _normalize_columns(columns)
    candidate_groups = [SST8_COLUMN_CANDIDATES] if preferred.lower() == "sst8" else [SST3_COLUMN_CANDIDATES]
    for group in candidate_groups:
        for candidate in group:
            if candidate in normalized:
                return normalized[candidate]
    for normalized_name, original_name in normalized.items():
        if preferred.lower() in normalized_name:
            return original_name
    raise ValueError(f"Could not discover the {preferred} target column.")


def discover_schema(dataframe: pd.DataFrame) -> dict[str, str]:
    """Discovers the columns used by the project."""

    columns = list(dataframe.columns)
    schema = {
        "sequence_column": discover_sequence_column(columns),
        "target_column": discover_target_column(columns, preferred="sst8"),
    }
    try:
        schema["auxiliary_target_column"] = discover_target_column(columns, preferred="sst3")
    except ValueError:
        schema["auxiliary_target_column"] = ""
    return schema


def load_named_splits(split_files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Loads all named dataset splits."""

    return {split_name: load_csv(path) for split_name, path in split_files.items()}
