from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class AminoAcidVocab:
    """Maps amino-acid characters to integer ids."""

    token_to_idx: dict[str, int]
    idx_to_token: list[str]
    pad_token: str = PAD_TOKEN
    unk_token: str = UNK_TOKEN

    @classmethod
    def from_sequences(cls, sequences: pd.Series) -> "AminoAcidVocab":
        unique_tokens = sorted(set("".join(sequences.tolist())))
        idx_to_token = [PAD_TOKEN, UNK_TOKEN] + unique_tokens
        token_to_idx = {token: index for index, token in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[self.unk_token]

    def encode(self, sequence: str) -> list[int]:
        return [self.token_to_idx.get(token, self.unk_idx) for token in sequence]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.idx_to_token[token_id] for token_id in token_ids if token_id != self.pad_idx)


@dataclass
class LabelEncoder:
    """Maps SST8 labels to integer ids."""

    label_to_idx: dict[str, int]
    idx_to_label: list[str]
    ignore_index: int = -100

    @classmethod
    def from_labels(cls, labels: pd.Series, ignore_index: int = -100) -> "LabelEncoder":
        unique_labels = sorted(set("".join(labels.tolist())))
        idx_to_label = unique_labels
        label_to_idx = {label: index for index, label in enumerate(idx_to_label)}
        return cls(label_to_idx=label_to_idx, idx_to_label=idx_to_label, ignore_index=ignore_index)

    def encode(self, labels: str) -> list[int]:
        return [self.label_to_idx[label] for label in labels]

    def decode(self, label_ids: list[int]) -> str:
        return "".join(self.idx_to_label[label_id] for label_id in label_ids if label_id != self.ignore_index)

    @property
    def labels(self) -> list[str]:
        return self.idx_to_label


def build_preprocessors(
    train_dataframe: pd.DataFrame,
    sequence_column: str,
    target_column: str,
    ignore_index: int,
) -> tuple[AminoAcidVocab, LabelEncoder]:
    """Builds the amino-acid vocabulary and SST8 label encoder from the training split."""

    vocab = AminoAcidVocab.from_sequences(train_dataframe[sequence_column])
    label_encoder = LabelEncoder.from_labels(train_dataframe[target_column], ignore_index=ignore_index)
    return vocab, label_encoder
