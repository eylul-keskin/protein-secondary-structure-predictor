"""Evaluate an existing SST8 checkpoint without training or replacing old outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import time


EVALUATION_SPLITS = ("validation", "cb513", "ts115", "casp12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True, help="A NEW directory; existing paths are rejected.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: PROJECT_ROOT/outputs/models/bilstm_best.pt")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Maximum evaluation runtime; no training is performed.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    """Keep personal absolute paths out of saved metadata."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def write_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def metadata_comparison(fresh: dict, original: dict) -> dict:
    """Compare metrics explicitly; permit tiny numeric/environment differences."""
    differences = {}
    comparisons = {}
    for key in ("accuracy", "macro_f1", "weighted_f1", "loss"):
        tolerance = 1e-5 if key == "loss" else 1e-6
        fresh_value = float(fresh[key])
        original_value = float(original[key])
        within_tolerance = math.isclose(fresh_value, original_value, rel_tol=0.0, abs_tol=tolerance)
        comparisons[key] = {
            "fresh": fresh_value,
            "checkpoint_metadata": original_value,
            "absolute_difference": abs(fresh_value - original_value),
            "absolute_tolerance": tolerance,
            "within_tolerance": within_tolerance,
        }
        if not within_tolerance:
            differences[key] = comparisons[key]
    for key in ("num_sequences", "num_tokens", "label_order"):
        if fresh[key] != original[key]:
            differences[key] = {"fresh": fresh[key], "checkpoint_metadata": original[key]}
    return {"passed": not differences, "differences": differences, "comparisons": comparisons,
            "structural_fields_checked_exactly": ["num_sequences", "num_tokens", "label_order"]}


def logged_batches(dataloader, split_name: str):
    total_batches = len(dataloader)
    for batch_index, batch in enumerate(dataloader, start=1):
        yield batch
        if batch_index % 25 == 0 or batch_index == total_batches:
            print(f"[bilstm/{split_name}] batch {batch_index}/{total_batches}", flush=True)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_dir.resolve()
    checkpoint_path = (args.checkpoint or project_root / "outputs/models/bilstm_best.pt").resolve()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if output_root.exists():
        raise FileExistsError("--output-dir must not exist; previous results will not be overwritten.")
    if not (project_root / "src").is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("Project src/ or the requested checkpoint could not be found.")

    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("figures", "metrics", "predictions", "logs"):
        (output_root / name).mkdir()
    os.environ["MPLCONFIGDIR"] = str(output_root / ".mpl-cache")
    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, str(project_root))
    faulthandler.dump_traceback_later(args.timeout_seconds, exit=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_timer = time.perf_counter()
    status_path = output_root / "logs/evaluation_summary.json"
    summary = {
        "evaluation_only": True,
        "training_executed": False,
        "status": "running",
        "started_at_utc": started_at,
        "device": "cpu",
        "torch_threads": 2,
        "actual_training_epoch_count": None,
        "training_epoch_note": "The checkpoint epochs field is a configured limit, not evidence of completed epochs.",
    }
    write_json(summary, status_path)

    try:
        import pandas as pd
        import torch

        from src.analysis import plot_class_distribution, plot_sequence_length_distribution, plot_test_set_comparison
        from src.config import TrainingConfig, build_config
        from src.data_loading import discover_schema, load_named_splits
        from src.data_validation import compute_split_overlaps, validate_and_clean_split
        from src.dataset import ProteinSequenceDataset, create_dataloader
        from src.evaluate import evaluate_predictions, predict_with_baseline, predict_with_model, save_evaluation_outputs
        from src.models import BiLSTMTagger, MostFrequentClassBaseline
        from src.preprocessing import build_preprocessors
        from src.utils import save_json

        torch.set_num_threads(2)
        config = build_config(project_root)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_config = checkpoint["training_config"]
        training_config = TrainingConfig(**checkpoint_config)
        if training_config.batch_size <= 0:
            raise ValueError("The checkpoint batch size must be positive.")
        summary.update({
            "training_config": checkpoint_config,
            "source_checkpoint": {
                "path": display_path(checkpoint_path, project_root),
                "sha256": sha256_file(checkpoint_path),
                "training_configuration": checkpoint_config,
                "original_best_validation_metrics": checkpoint["best_validation_metrics"],
            },
            "versions": {"python": sys.version.split()[0], **{
                package: metadata.version(package)
                for package in ("torch", "pandas", "numpy", "scikit-learn", "matplotlib")
            }},
            "evaluation_settings": {"batch_size": training_config.batch_size, "num_workers": 0, "seed_not_used_for_training": True},
            "input_files": {
                split: {"path": display_path(path, project_root), "sha256": sha256_file(path)}
                for split, path in config.paths.split_files.items()
            },
            "source_files_sha256": {
                path.relative_to(project_root).as_posix(): sha256_file(path)
                for path in sorted((project_root / "src").glob("*.py"))
            },
            "evaluation_script_sha256": sha256_file(Path(__file__)),
        })
        write_json(summary, status_path)

        print("Loading and validating the five provided CSV splits...", flush=True)
        raw_splits = load_named_splits(config.paths.split_files)
        schema = discover_schema(raw_splits["train"])
        sequence_column, target_column = schema["sequence_column"], schema["target_column"]
        cleaned_splits, validation_summaries = {}, {}
        train_frame, train_summary = validate_and_clean_split(
            raw_splits["train"], "train", Path(config.paths.train_csv.name), sequence_column, target_column,
        )
        if train_frame.empty:
            raise ValueError("No usable training rows remain for vocabulary/baseline construction.")
        cleaned_splits["train"] = train_frame
        validation_summaries["train"] = train_summary.to_dict()
        allowed_labels = set(train_summary.unique_target_labels)
        for split in EVALUATION_SPLITS:
            frame, split_summary = validate_and_clean_split(
                raw_splits[split], split, Path(config.paths.split_files[split].name), sequence_column, target_column, allowed_labels,
            )
            if frame.empty:
                raise ValueError(f"No usable rows remain in {split}.")
            cleaned_splits[split] = frame
            validation_summaries[split] = split_summary.to_dict()
        save_json({
            "schema": schema,
            "split_validation": validation_summaries,
            "split_overlaps": compute_split_overlaps(cleaned_splits, sequence_column),
        }, output_root / "logs/data_validation_summary.json")

        vocab, encoder = build_preprocessors(train_frame, sequence_column, target_column, training_config.ignore_index)
        preprocessing = {
            "vocab_size": len(vocab.idx_to_token), "vocab_tokens": vocab.idx_to_token,
            "label_to_idx": encoder.label_to_idx, "idx_to_label": encoder.idx_to_label,
            "pad_idx": vocab.pad_idx, "unk_idx": vocab.unk_idx, "ignore_index": encoder.ignore_index,
        }
        saved_preprocessing_path = project_root / "outputs/logs/preprocessing_summary.json"
        saved_preprocessing = json.loads(saved_preprocessing_path.read_text(encoding="utf-8-sig"))
        if preprocessing != saved_preprocessing:
            raise ValueError("Training-derived preprocessing does not match the saved preprocessing artifact.")
        if checkpoint["label_to_idx"] != encoder.label_to_idx or checkpoint["idx_to_label"] != encoder.idx_to_label:
            raise ValueError("The checkpoint labels do not match the training-derived encoder.")
        save_json(preprocessing, output_root / "logs/preprocessing_summary.json")
        summary["preprocessing_matches_saved_artifact"] = True
        summary["schema"] = schema

        baseline = MostFrequentClassBaseline()
        baseline.fit(train_frame[target_column].tolist())
        model = BiLSTMTagger(
            vocab_size=len(vocab.idx_to_token), embedding_dim=training_config.embedding_dim,
            hidden_dim=training_config.hidden_dim, num_labels=len(encoder.labels), pad_idx=vocab.pad_idx,
            num_layers=training_config.num_layers, dropout=training_config.dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        summary["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        summary["baseline_majority_label"] = baseline.majority_label
        summary["baseline_results"], summary["bilstm_results"] = {}, {}
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=training_config.ignore_index)
        comparison_rows = []

        for split in EVALUATION_SPLITS:
            frame = cleaned_splits[split]
            print(f"[baseline/{split}] evaluating {len(frame)} sequences", flush=True)
            records, true_labels, predictions = predict_with_baseline(baseline, frame, sequence_column, target_column)
            baseline_bundle = evaluate_predictions(
                records, true_labels, predictions, encoder.labels, split, "baseline",
                tuple(training_config.length_bucket_edges), training_config.sample_prediction_count,
            )
            save_evaluation_outputs(baseline_bundle, output_root / "metrics", output_root / "figures", output_root / "predictions")
            summary["baseline_results"][split] = baseline_bundle.metrics

            print(f"[bilstm/{split}] evaluating {len(frame)} sequences once on CPU", flush=True)
            dataset = ProteinSequenceDataset(frame, sequence_column, target_column, vocab, encoder)
            dataloader = create_dataloader(dataset, training_config.batch_size, False, vocab.pad_idx, encoder.ignore_index, num_workers=0)
            records, true_labels, predictions, average_loss = predict_with_model(
                model, logged_batches(dataloader, split), encoder, torch.device("cpu"), loss_function,
            )
            bilstm_bundle = evaluate_predictions(
                records, true_labels, predictions, encoder.labels, split, "bilstm",
                tuple(training_config.length_bucket_edges), training_config.sample_prediction_count, average_loss,
            )
            if not all(math.isfinite(float(bilstm_bundle.metrics[key])) for key in ("accuracy", "macro_f1", "weighted_f1", "loss")):
                raise ValueError(f"Non-finite evaluation metric in {split}.")
            save_evaluation_outputs(bilstm_bundle, output_root / "metrics", output_root / "figures", output_root / "predictions")
            summary["bilstm_results"][split] = bilstm_bundle.metrics
            for bundle in (baseline_bundle, bilstm_bundle):
                comparison_rows.append({key: bundle.metrics[key] for key in ("model", "split", "accuracy", "macro_f1", "weighted_f1")})
            if split == "validation":
                summary["checkpoint_validation_metadata_check"] = metadata_comparison(
                    bilstm_bundle.metrics, checkpoint["best_validation_metrics"],
                )
                write_json(summary, status_path)
                if not summary["checkpoint_validation_metadata_check"]["passed"]:
                    raise ValueError("Fresh validation metrics differ from the checkpoint metadata; inspect the evaluation summary before further evaluation.")
            write_json(summary, status_path)
            print(f"[{split}] complete: accuracy={bilstm_bundle.metrics['accuracy']:.8f}; macro_f1={bilstm_bundle.metrics['macro_f1']:.8f}", flush=True)

        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(output_root / "metrics/model_comparison.csv", index=False)
        test_rows = []
        for split in ("cb513", "ts115", "casp12"):
            row = {"split": split}
            for model_name in ("baseline", "bilstm"):
                for key in ("accuracy", "macro_f1", "weighted_f1"):
                    row[f"{model_name}_{key}"] = summary[f"{model_name}_results"][split][key]
            test_rows.append(row)
        pd.DataFrame(test_rows).to_csv(output_root / "metrics/test_set_comparison.csv", index=False)
        plot_test_set_comparison(comparison, output_root / "figures/test_set_comparison.png")
        plot_class_distribution({split: row["label_distribution"] for split, row in validation_summaries.items()}, output_root / "figures/class_distribution.png")
        plot_sequence_length_distribution({split: frame[sequence_column].str.len().tolist() for split, frame in cleaned_splits.items()}, output_root / "figures/sequence_length_distribution.png")
        if sha256_file(checkpoint_path) != summary["source_checkpoint"]["sha256"] or any(
            sha256_file(path) != summary["input_files"][split]["sha256"]
            for split, path in config.paths.split_files.items()
        ):
            raise ValueError("An input CSV or checkpoint changed during evaluation.")
        summary["input_files_unchanged_during_evaluation"] = True
        summary["status"] = "completed"
        summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = round(time.perf_counter() - started_timer, 3)
        write_json(summary, status_path)
        print(json.dumps({"status": summary["status"], "checkpoint_validation_metadata_check": summary["checkpoint_validation_metadata_check"], "elapsed_seconds": summary["elapsed_seconds"], "bilstm_results": {split: {key: result[key] for key in ("num_sequences", "num_tokens", "accuracy", "macro_f1", "weighted_f1", "loss")} for split, result in summary["bilstm_results"].items()}}, ensure_ascii=False), flush=True)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error_type"] = type(error).__name__
        summary["elapsed_seconds"] = round(time.perf_counter() - started_timer, 3)
        write_json(summary, status_path)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
