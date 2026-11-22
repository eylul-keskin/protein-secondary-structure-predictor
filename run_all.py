from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis import (
    plot_class_distribution,
    plot_sequence_length_distribution,
    plot_test_set_comparison,
    plot_training_history,
)
from src.config import build_config
from src.data_loading import discover_schema, load_named_splits
from src.data_validation import compute_split_overlaps, validate_and_clean_split
from src.dataset import ProteinSequenceDataset, create_dataloader
from src.evaluate import evaluate_predictions, predict_with_baseline, predict_with_model, save_evaluation_outputs
from src.models import BiLSTMTagger, MostFrequentClassBaseline
from src.preprocessing import build_preprocessors
from src.train import train_model_with_encoder
from src.utils import ensure_directories, save_dataframe, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SST8 protein secondary-structure project pipeline.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the default batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device preference. 'auto' uses CUDA when available.",
    )
    return parser.parse_args()


def resolve_device(device_preference: str) -> torch.device:
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def summarize_schema(schema: dict[str, str]) -> dict[str, str]:
    return {
        "sequence_column": schema["sequence_column"],
        "target_column": schema["target_column"],
        "auxiliary_target_column": schema.get("auxiliary_target_column", ""),
    }


def main() -> None:
    args = parse_args()
    config = build_config(Path.cwd())
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    config.preferred_device = args.device

    ensure_directories(config.paths.output_dirs)
    set_seed(config.training.seed)

    raw_splits = load_named_splits(config.paths.split_files)
    schema = discover_schema(raw_splits["train"])
    config.sequence_column = schema["sequence_column"]
    config.target_column = schema["target_column"]
    if schema.get("auxiliary_target_column"):
        config.auxiliary_target_column = schema["auxiliary_target_column"]

    cleaned_splits: dict[str, pd.DataFrame] = {}
    validation_summaries = {}

    train_cleaned, train_summary = validate_and_clean_split(
        dataframe=raw_splits["train"],
        split_name="train",
        file_path=config.paths.train_csv,
        sequence_column=config.sequence_column,
        target_column=config.target_column,
        allowed_label_characters=None,
    )
    cleaned_splits["train"] = train_cleaned
    validation_summaries["train"] = train_summary.to_dict()

    allowed_labels = set(train_summary.unique_target_labels)
    for split_name in ("validation", "cb513", "ts115", "casp12"):
        cleaned_dataframe, split_summary = validate_and_clean_split(
            dataframe=raw_splits[split_name],
            split_name=split_name,
            file_path=config.paths.split_files[split_name],
            sequence_column=config.sequence_column,
            target_column=config.target_column,
            allowed_label_characters=allowed_labels,
        )
        cleaned_splits[split_name] = cleaned_dataframe
        validation_summaries[split_name] = split_summary.to_dict()

    overlaps = compute_split_overlaps(cleaned_splits, config.sequence_column)
    validation_payload = {
        "schema": summarize_schema(schema),
        "split_validation": validation_summaries,
        "split_overlaps": overlaps,
    }
    save_json(validation_payload, config.paths.logs_dir / "data_validation_summary.json")

    label_counts_by_split = {
        split_name: validation_summaries[split_name]["label_distribution"] for split_name in cleaned_splits
    }
    lengths_by_split = {
        split_name: cleaned_splits[split_name][config.sequence_column].str.len().tolist() for split_name in cleaned_splits
    }
    plot_class_distribution(label_counts_by_split, config.paths.figures_dir / "class_distribution.png")
    plot_sequence_length_distribution(lengths_by_split, config.paths.figures_dir / "sequence_length_distribution.png")

    vocab, label_encoder = build_preprocessors(
        train_dataframe=cleaned_splits["train"],
        sequence_column=config.sequence_column,
        target_column=config.target_column,
        ignore_index=config.training.ignore_index,
    )
    preprocessing_summary = {
        "vocab_size": len(vocab.idx_to_token),
        "vocab_tokens": vocab.idx_to_token,
        "label_to_idx": label_encoder.label_to_idx,
        "idx_to_label": label_encoder.idx_to_label,
        "pad_idx": vocab.pad_idx,
        "unk_idx": vocab.unk_idx,
        "ignore_index": label_encoder.ignore_index,
    }
    save_json(preprocessing_summary, config.paths.logs_dir / "preprocessing_summary.json")

    baseline = MostFrequentClassBaseline()
    baseline.fit(cleaned_splits["train"][config.target_column].tolist())

    baseline_results = {}
    comparison_rows: list[dict[str, object]] = []
    for split_name in ("validation", "cb513", "ts115", "casp12"):
        prediction_records, all_true, all_predicted = predict_with_baseline(
            baseline=baseline,
            dataframe=cleaned_splits[split_name],
            sequence_column=config.sequence_column,
            target_column=config.target_column,
        )
        bundle = evaluate_predictions(
            prediction_records=prediction_records,
            all_true=all_true,
            all_predicted=all_predicted,
            label_names=label_encoder.labels,
            split_name=split_name,
            model_name="baseline",
            bucket_edges=config.training.length_bucket_edges,
            sample_count=config.training.sample_prediction_count,
        )
        save_evaluation_outputs(bundle, config.paths.metrics_dir, config.paths.figures_dir, config.paths.predictions_dir)
        baseline_results[split_name] = bundle.metrics
        comparison_rows.append(
            {
                "model": "baseline",
                "split": split_name,
                "accuracy": bundle.metrics["accuracy"],
                "macro_f1": bundle.metrics["macro_f1"],
                "weighted_f1": bundle.metrics["weighted_f1"],
            }
        )

    train_dataset = ProteinSequenceDataset(
        dataframe=cleaned_splits["train"],
        sequence_column=config.sequence_column,
        target_column=config.target_column,
        vocab=vocab,
        label_encoder=label_encoder,
    )
    validation_dataset = ProteinSequenceDataset(
        dataframe=cleaned_splits["validation"],
        sequence_column=config.sequence_column,
        target_column=config.target_column,
        vocab=vocab,
        label_encoder=label_encoder,
    )
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pad_idx=vocab.pad_idx,
        ignore_index=config.training.ignore_index,
        num_workers=config.training.num_workers,
    )
    validation_loader = create_dataloader(
        dataset=validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        pad_idx=vocab.pad_idx,
        ignore_index=config.training.ignore_index,
        num_workers=config.training.num_workers,
    )

    device = resolve_device(config.preferred_device)
    model = BiLSTMTagger(
        vocab_size=len(vocab.idx_to_token),
        embedding_dim=config.training.embedding_dim,
        hidden_dim=config.training.hidden_dim,
        num_labels=len(label_encoder.labels),
        pad_idx=vocab.pad_idx,
        num_layers=config.training.num_layers,
        dropout=config.training.dropout,
    ).to(device)

    checkpoint_path = config.paths.models_dir / "bilstm_best.pt"
    history = train_model_with_encoder(
        model=model,
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        label_encoder=label_encoder,
        device=device,
        checkpoint_path=checkpoint_path,
        training_config=config.training,
    )
    save_dataframe(history, config.paths.metrics_dir / "training_history.csv")
    plot_training_history(history, config.paths.figures_dir / "training_history.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bilstm_results = {}
    evaluation_loss_function = torch.nn.CrossEntropyLoss(ignore_index=config.training.ignore_index)
    for split_name in ("validation", "cb513", "ts115", "casp12"):
        dataset = ProteinSequenceDataset(
            dataframe=cleaned_splits[split_name],
            sequence_column=config.sequence_column,
            target_column=config.target_column,
            vocab=vocab,
            label_encoder=label_encoder,
        )
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            pad_idx=vocab.pad_idx,
            ignore_index=config.training.ignore_index,
            num_workers=config.training.num_workers,
        )
        prediction_records, all_true, all_predicted, average_loss = predict_with_model(
            model=model,
            dataloader=dataloader,
            label_encoder=label_encoder,
            device=device,
            loss_function=evaluation_loss_function,
        )
        bundle = evaluate_predictions(
            prediction_records=prediction_records,
            all_true=all_true,
            all_predicted=all_predicted,
            label_names=label_encoder.labels,
            split_name=split_name,
            model_name="bilstm",
            bucket_edges=config.training.length_bucket_edges,
            sample_count=config.training.sample_prediction_count,
            average_loss=average_loss,
        )
        save_evaluation_outputs(bundle, config.paths.metrics_dir, config.paths.figures_dir, config.paths.predictions_dir)
        bilstm_results[split_name] = bundle.metrics
        comparison_rows.append(
            {
                "model": "bilstm",
                "split": split_name,
                "accuracy": bundle.metrics["accuracy"],
                "macro_f1": bundle.metrics["macro_f1"],
                "weighted_f1": bundle.metrics["weighted_f1"],
            }
        )

    comparison_dataframe = pd.DataFrame(comparison_rows)
    save_dataframe(comparison_dataframe, config.paths.metrics_dir / "model_comparison.csv")

    test_set_comparison_rows = []
    for split_name in ("cb513", "ts115", "casp12"):
        test_set_comparison_rows.append(
            {
                "split": split_name,
                "baseline_accuracy": baseline_results[split_name]["accuracy"],
                "baseline_macro_f1": baseline_results[split_name]["macro_f1"],
                "baseline_weighted_f1": baseline_results[split_name]["weighted_f1"],
                "bilstm_accuracy": bilstm_results[split_name]["accuracy"],
                "bilstm_macro_f1": bilstm_results[split_name]["macro_f1"],
                "bilstm_weighted_f1": bilstm_results[split_name]["weighted_f1"],
            }
        )
    test_set_comparison = pd.DataFrame(test_set_comparison_rows)
    save_dataframe(test_set_comparison, config.paths.metrics_dir / "test_set_comparison.csv")
    plot_test_set_comparison(comparison_dataframe, config.paths.figures_dir / "test_set_comparison.png")

    run_summary = {
        "device": str(device),
        "training_config": vars(config.training),
        "schema": summarize_schema(schema),
        "baseline_results": baseline_results,
        "bilstm_results": bilstm_results,
    }
    save_json(run_summary, config.paths.logs_dir / "run_summary.json")


if __name__ == "__main__":
    main()
