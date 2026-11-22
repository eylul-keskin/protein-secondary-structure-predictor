from __future__ import annotations

from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MostFrequentClassBaseline:
    """Predicts the single most common SST8 label for every token."""

    def __init__(self) -> None:
        self.majority_label: str | None = None

    def fit(self, label_strings: list[str]) -> None:
        counts = Counter("".join(label_strings))
        self.majority_label = counts.most_common(1)[0][0]

    def predict(self, sequences: list[str]) -> list[str]:
        if self.majority_label is None:
            raise RuntimeError("The baseline must be fitted before prediction.")
        return [self.majority_label * len(sequence) for sequence in sequences]


class BiLSTMTagger(nn.Module):
    """Embedding + BiLSTM + linear head for SST8 token classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        pad_idx: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        return self.classifier(self.dropout(outputs))
