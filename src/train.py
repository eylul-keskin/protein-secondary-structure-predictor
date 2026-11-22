from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .evaluate import evaluate_predictions, predict_with_model


def train_model_with_encoder(
    model: torch.nn.Module,
    train_dataloader,
    validation_dataloader,
    label_encoder,
    device: torch.device,
    checkpoint_path: Path,
    training_config,
) -> pd.DataFrame:
    """Trains the BiLSTM and keeps the checkpoint with the best validation macro F1."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=training_config.ignore_index)

    history_rows: list[dict[str, float | int]] = []
    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_token_total = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()

            token_count = int(batch["mask"].sum().item())
            train_loss_total += float(loss.item()) * token_count
            train_token_total += token_count

        train_loss = train_loss_total / max(train_token_total, 1)
        validation_records, validation_true, validation_predicted, validation_loss = predict_with_model(
            model=model,
            dataloader=validation_dataloader,
            label_encoder=label_encoder,
            device=device,
            loss_function=loss_function,
        )
        validation_bundle = evaluate_predictions(
            prediction_records=validation_records,
            all_true=validation_true,
            all_predicted=validation_predicted,
            label_names=label_encoder.labels,
            split_name="validation",
            model_name="bilstm",
            bucket_edges=training_config.length_bucket_edges,
            sample_count=training_config.sample_prediction_count,
            average_loss=validation_loss,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_bundle.metrics.get("loss", float("nan")),
                "val_accuracy": validation_bundle.metrics["accuracy"],
                "val_macro_f1": validation_bundle.metrics["macro_f1"],
                "val_weighted_f1": validation_bundle.metrics["weighted_f1"],
            }
        )

        if validation_bundle.metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(validation_bundle.metrics["macro_f1"])
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_validation_metrics": validation_bundle.metrics,
                "training_config": asdict(training_config),
                "label_to_idx": label_encoder.label_to_idx,
                "idx_to_label": label_encoder.idx_to_label,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                break

    return pd.DataFrame(history_rows)
