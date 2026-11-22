from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .preprocessing import AminoAcidVocab, LabelEncoder


class ProteinSequenceDataset(Dataset):
    """Stores encoded sequences and labels for token-level prediction."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_column: str,
        target_column: str,
        vocab: AminoAcidVocab,
        label_encoder: LabelEncoder,
    ) -> None:
        working = dataframe[[sequence_column, target_column]].reset_index(drop=True)
        self.records: list[dict[str, Any]] = []
        for record_id, row in working.iterrows():
            sequence = row[sequence_column]
            labels = row[target_column]
            self.records.append(
                {
                    "record_id": int(record_id),
                    "sequence": sequence,
                    "label_string": labels,
                    "length": len(sequence),
                    "input_ids": torch.tensor(vocab.encode(sequence), dtype=torch.long),
                    "label_ids": torch.tensor(label_encoder.encode(labels), dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_protein_batch(
    batch: list[dict[str, Any]],
    pad_idx: int,
    ignore_index: int,
) -> dict[str, Any]:
    """Pads sequences and labels and produces a token mask."""

    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_idx)
    label_ids = pad_sequence([item["label_ids"] for item in batch], batch_first=True, padding_value=ignore_index)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = label_ids != ignore_index
    return {
        "record_ids": [item["record_id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "label_strings": [item["label_string"] for item in batch],
        "input_ids": input_ids,
        "labels": label_ids,
        "lengths": lengths,
        "mask": mask,
    }


def create_dataloader(
    dataset: ProteinSequenceDataset,
    batch_size: int,
    shuffle: bool,
    pad_idx: int,
    ignore_index: int,
    num_workers: int = 0,
) -> DataLoader:
    """Builds a DataLoader with the project-specific collate function."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_protein_batch(batch, pad_idx=pad_idx, ignore_index=ignore_index),
    )
