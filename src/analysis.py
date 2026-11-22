from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(label_counts_by_split: dict[str, dict[str, int]], save_path: Path) -> None:
    """Plots the SST8 class distribution across splits as percentages."""

    dataframe = pd.DataFrame(label_counts_by_split).T.fillna(0)
    percentages = dataframe.div(dataframe.sum(axis=1), axis=0) * 100.0

    figure, axis = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(percentages))
    for label in percentages.columns:
        values = percentages[label].values
        axis.bar(percentages.index, values, bottom=bottom, label=label)
        bottom += values

    axis.set_title("SST8 Class Distribution by Split")
    axis.set_ylabel("Percentage of Tokens")
    axis.set_xlabel("Dataset Split")
    axis.legend(title="SST8 Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_ylim(0, 100)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_sequence_length_distribution(lengths_by_split: dict[str, list[int]], save_path: Path) -> None:
    """Plots the sequence length distribution for each split."""

    figure, axis = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(max(lengths) for lengths in lengths_by_split.values()) + 25, 30)
    for split_name, lengths in lengths_by_split.items():
        axis.hist(lengths, bins=bins, alpha=0.45, density=True, label=split_name)

    axis.set_title("Sequence Length Distribution by Split")
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_training_history(history: pd.DataFrame, save_path: Path) -> None:
    """Plots training loss and validation metrics over epochs."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="Validation Loss")
    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_accuracy"], marker="o", label="Validation Accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], marker="o", label="Validation Macro F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(confusion: pd.DataFrame, save_path: Path, title: str) -> None:
    """Plots a row-normalized confusion matrix for easier interpretation."""

    matrix = confusion.values.astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(confusion.columns)))
    axis.set_xticklabels(confusion.columns)
    axis.set_yticks(range(len(confusion.index)))
    axis.set_yticklabels(confusion.index)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title(title)

    for row_index in range(normalized.shape[0]):
        for col_index in range(normalized.shape[1]):
            value = normalized[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized proportion")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_test_set_comparison(comparison: pd.DataFrame, save_path: Path) -> None:
    """Plots macro F1 across the benchmark splits for baseline vs BiLSTM."""

    test_rows = comparison[comparison["split"].isin(["cb513", "ts115", "casp12"])]
    pivot = test_rows.pivot(index="split", columns="model", values="macro_f1")
    figure, axis = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Macro F1 Across Test Sets")
    axis.set_xlabel("Test Split")
    axis.set_ylabel("Macro F1")
    axis.legend(title="Model")
    axis.set_ylim(0.0, min(1.0, max(0.05, float(pivot.max().max()) + 0.05)))
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
