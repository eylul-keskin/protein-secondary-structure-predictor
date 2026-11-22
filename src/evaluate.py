from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .analysis import plot_confusion_matrix
from .models import MostFrequentClassBaseline
from .preprocessing import LabelEncoder
from .utils import save_dataframe, save_json


@dataclass
class EvaluationBundle:
    """Holds the outputs saved for one evaluated split."""

    split_name: str
    model_name: str
    metrics: dict
    per_class_report: pd.DataFrame
    confusion: pd.DataFrame
    top_confusions: pd.DataFrame
    length_bucket_report: pd.DataFrame
    sample_predictions: pd.DataFrame
    sequence_level_results: pd.DataFrame


def _build_prediction_record(record_id: int, sequence: str, true_labels: str, predicted_labels: str) -> dict[str, object]:
    mismatches = [index + 1 for index, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)) if true_label != predicted_label]
    error_count = len(mismatches)
    token_accuracy = 1.0 - (error_count / len(true_labels))
    mismatch_preview = ",".join(str(index) for index in mismatches[:20])
    return {
        "record_id": record_id,
        "length": len(sequence),
        "token_accuracy": round(token_accuracy, 4),
        "error_count": error_count,
        "mismatch_positions_preview": mismatch_preview,
        "sequence": sequence,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
    }


def predict_with_model(
    model: torch.nn.Module,
    dataloader,
    label_encoder: LabelEncoder,
    device: torch.device,
    loss_function: torch.nn.Module | None = None,
) -> tuple[list[dict[str, object]], list[str], list[str], float | None]:
    """Runs model inference and returns per-sequence prediction records."""

    model.eval()
    prediction_records: list[dict[str, object]] = []
    all_true_labels: list[str] = []
    all_predicted_labels: list[str] = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            logits = model(input_ids, lengths)
            predictions = logits.argmax(dim=-1).cpu()

            if loss_function is not None:
                loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
                token_count = int(batch["mask"].sum().item())
                total_loss += float(loss.item()) * token_count
                total_tokens += token_count

            lengths_cpu = batch["lengths"].tolist()
            for item_index, sequence_length in enumerate(lengths_cpu):
                true_labels = batch["label_strings"][item_index]
                predicted_labels = label_encoder.decode(predictions[item_index][:sequence_length].tolist())
                prediction_records.append(
                    _build_prediction_record(
                        record_id=batch["record_ids"][item_index],
                        sequence=batch["sequences"][item_index],
                        true_labels=true_labels,
                        predicted_labels=predicted_labels,
                    )
                )
                all_true_labels.extend(list(true_labels))
                all_predicted_labels.extend(list(predicted_labels))

    average_loss = (total_loss / total_tokens) if total_tokens else None
    return prediction_records, all_true_labels, all_predicted_labels, average_loss


def predict_with_baseline(
    baseline: MostFrequentClassBaseline,
    dataframe: pd.DataFrame,
    sequence_column: str,
    target_column: str,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    """Runs the baseline on a DataFrame and returns per-sequence prediction records."""

    prediction_records: list[dict[str, object]] = []
    all_true_labels: list[str] = []
    all_predicted_labels: list[str] = []

    sequences = dataframe[sequence_column].tolist()
    predicted_sequences = baseline.predict(sequences)
    for record_id, row in enumerate(dataframe[[sequence_column, target_column]].itertuples(index=False)):
        sequence = row[0]
        true_labels = row[1]
        predicted_labels = predicted_sequences[record_id]
        prediction_records.append(
            _build_prediction_record(
                record_id=record_id,
                sequence=sequence,
                true_labels=true_labels,
                predicted_labels=predicted_labels,
            )
        )
        all_true_labels.extend(list(true_labels))
        all_predicted_labels.extend(list(predicted_labels))
    return prediction_records, all_true_labels, all_predicted_labels


def _build_per_class_report(all_true: list[str], all_predicted: list[str], label_names: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true,
        all_predicted,
        labels=label_names,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def _build_top_confusions(confusion: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for true_label in confusion.index:
        for predicted_label in confusion.columns:
            if true_label == predicted_label:
                continue
            count = int(confusion.loc[true_label, predicted_label])
            if count > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count"])
    return result.sort_values("count", ascending=False).reset_index(drop=True)


def _assign_bucket(length: int, bucket_edges: tuple[int, int]) -> str:
    short_edge, medium_edge = bucket_edges
    if length <= short_edge:
        return f"short_<=_{short_edge}"
    if length <= medium_edge:
        return f"medium_{short_edge + 1}_{medium_edge}"
    return f"long_>_{medium_edge}"


def _build_length_bucket_report(
    sequence_level_results: pd.DataFrame,
    label_names: list[str],
    bucket_edges: tuple[int, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if sequence_level_results.empty:
        return pd.DataFrame(rows)

    bucketed = sequence_level_results.copy()
    bucketed["length_bucket"] = bucketed["length"].apply(lambda value: _assign_bucket(value, bucket_edges))

    for bucket_name, group in bucketed.groupby("length_bucket", sort=False):
        all_true = [label for labels in group["true_labels"] for label in labels]
        all_predicted = [label for labels in group["predicted_labels"] for label in labels]
        rows.append(
            {
                "length_bucket": bucket_name,
                "num_sequences": int(len(group)),
                "num_tokens": int(sum(group["length"])),
                "accuracy": float(accuracy_score(all_true, all_predicted)),
                "macro_f1": float(f1_score(all_true, all_predicted, labels=label_names, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(all_true, all_predicted, labels=label_names, average="weighted", zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_sample_predictions(sequence_level_results: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if sequence_level_results.empty:
        return sequence_level_results

    ordered_by_error = sequence_level_results.sort_values(
        by=["token_accuracy", "error_count", "length"],
        ascending=[True, False, False],
    )
    ordered_by_length = sequence_level_results.sort_values("length")

    candidate_indices: list[int] = []
    candidate_indices.extend(ordered_by_error.head(3).index.tolist())
    candidate_indices.extend(ordered_by_error.tail(2).index.tolist())
    candidate_indices.extend(ordered_by_length.head(1).index.tolist())
    candidate_indices.extend(ordered_by_length.tail(1).index.tolist())
    candidate_indices.append(ordered_by_error.index[len(ordered_by_error) // 2])

    selected: list[int] = []
    for index in candidate_indices:
        if index not in selected:
            selected.append(index)
    for index in ordered_by_error.index.tolist():
        if len(selected) >= sample_count:
            break
        if index not in selected:
            selected.append(index)

    return sequence_level_results.loc[selected].head(sample_count).reset_index(drop=True)


def evaluate_predictions(
    prediction_records: list[dict[str, object]],
    all_true: list[str],
    all_predicted: list[str],
    label_names: list[str],
    split_name: str,
    model_name: str,
    bucket_edges: tuple[int, int],
    sample_count: int,
    average_loss: float | None = None,
) -> EvaluationBundle:
    """Computes token-level metrics and report-friendly artifacts."""

    per_class_report = _build_per_class_report(all_true, all_predicted, label_names)
    confusion = pd.DataFrame(
        confusion_matrix(all_true, all_predicted, labels=label_names),
        index=label_names,
        columns=label_names,
    )
    top_confusions = _build_top_confusions(confusion)
    sequence_level_results = pd.DataFrame(prediction_records).sort_values("record_id").reset_index(drop=True)
    length_bucket_report = _build_length_bucket_report(sequence_level_results, label_names, bucket_edges)
    sample_predictions = _select_sample_predictions(sequence_level_results, sample_count)

    metrics = {
        "split": split_name,
        "model": model_name,
        "num_sequences": int(len(sequence_level_results)),
        "num_tokens": int(len(all_true)),
        "accuracy": float(accuracy_score(all_true, all_predicted)),
        "macro_f1": float(f1_score(all_true, all_predicted, labels=label_names, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(all_true, all_predicted, labels=label_names, average="weighted", zero_division=0)
        ),
        "label_order": label_names,
        "top_confusion_pairs": top_confusions.head(10).to_dict(orient="records"),
    }
    if average_loss is not None:
        metrics["loss"] = float(average_loss)

    return EvaluationBundle(
        split_name=split_name,
        model_name=model_name,
        metrics=metrics,
        per_class_report=per_class_report,
        confusion=confusion,
        top_confusions=top_confusions,
        length_bucket_report=length_bucket_report,
        sample_predictions=sample_predictions,
        sequence_level_results=sequence_level_results,
    )


def save_evaluation_outputs(
    bundle: EvaluationBundle,
    metrics_dir: Path,
    figures_dir: Path,
    predictions_dir: Path,
) -> None:
    """Saves metrics tables, JSON, confusion matrix, and sample predictions."""

    split_name = bundle.split_name
    model_name = bundle.model_name
    main_model = model_name == "bilstm"

    metrics_json_path = metrics_dir / f"{model_name}_{split_name}_metrics.json"
    per_class_path = metrics_dir / (
        f"per_class_report_{split_name}.csv" if main_model else f"{model_name}_per_class_report_{split_name}.csv"
    )
    confusion_csv_path = metrics_dir / (
        f"confusion_matrix_{split_name}.csv" if main_model else f"{model_name}_confusion_matrix_{split_name}.csv"
    )
    top_confusions_path = metrics_dir / (
        f"top_confusions_{split_name}.csv" if main_model else f"{model_name}_top_confusions_{split_name}.csv"
    )
    length_bucket_path = metrics_dir / (
        f"length_bucket_performance_{split_name}.csv"
        if main_model
        else f"{model_name}_length_bucket_performance_{split_name}.csv"
    )
    confusion_figure_path = figures_dir / (
        f"confusion_matrix_{split_name}.png" if main_model else f"{model_name}_confusion_matrix_{split_name}.png"
    )
    sample_predictions_path = predictions_dir / (
        f"sample_predictions_{split_name}.csv" if main_model else f"{model_name}_sample_predictions_{split_name}.csv"
    )

    save_json(bundle.metrics, metrics_json_path)
    save_dataframe(bundle.per_class_report, per_class_path)
    save_dataframe(bundle.confusion.reset_index().rename(columns={"index": "true_label"}), confusion_csv_path)
    save_dataframe(bundle.top_confusions, top_confusions_path)
    save_dataframe(bundle.length_bucket_report, length_bucket_path)
    save_dataframe(bundle.sample_predictions, sample_predictions_path)
    plot_confusion_matrix(
        bundle.confusion,
        confusion_figure_path,
        title=f"{model_name.upper()} Confusion Matrix ({split_name})",
    )
