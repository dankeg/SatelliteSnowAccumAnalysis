from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SatelliteSnowAccumAnalysis.Training.ImageDataset import ImageDataset
from SatelliteSnowAccumAnalysis.Training.cnn_segmentation import SnowCNN


REPO_ROOT = Path(__file__).resolve().parents[2]

IGNORE_LABEL = 255
TRAIN_MANIFEST = REPO_ROOT / "snow_dataset_small" / "manifests" / "train.csv"
VAL_MANIFEST = REPO_ROOT / "snow_dataset_small" / "manifests" / "val.csv"
TRAIN_STATS = REPO_ROOT / "snow_dataset_small" / "metadata" / "train_channel_stats.json"
OUTPUT_DIR = REPO_ROOT / "cnn_training_outputs"

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

METRICS = ["loss", "accuracy", "precision", "recall", "f1", "iou"]


def make_history() -> dict[str, list[float]]:
    history = {}
    for split in ("train", "val"):
        for metric in METRICS:
            history[f"{split}_{metric}"] = []
    return history


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    valid = labels != IGNORE_LABEL
    preds = preds[valid]
    labels = labels[valid]

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = nullcontext() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if not torch.any(labels != IGNORE_LABEL):
                continue

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(outputs.argmax(dim=1).detach().cpu())
            all_labels.append(labels.detach().cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    metrics = compute_metrics(preds, labels)
    metrics["loss"] = total_loss / len(all_preds)

    return metrics


def add_to_history(
    history: dict[str, list[float]],
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    for metric in METRICS:
        history[f"train_{metric}"].append(train_metrics[metric])
        history[f"val_{metric}"].append(val_metrics[metric])


def save_plot(
    history: dict[str, list[float]],
    keys: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    for key, label in zip(keys, labels):
        plt.plot(epochs, history[key], label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def plot_history(history: dict[str, list[float]]) -> None:
    save_plot(
        history,
        ["train_loss", "val_loss"],
        ["train", "val"],
        "Loss",
        "Loss vs Epoch",
        "loss_curve.png",
    )

    save_plot(
        history,
        ["train_accuracy", "val_accuracy", "train_f1", "val_f1", "train_iou", "val_iou"],
        ["train accuracy", "val accuracy", "train f1", "val f1", "train iou", "val iou"],
        "Score",
        "Segmentation Metrics vs Epoch",
        "metrics_curve.png",
    )

    save_plot(
        history,
        ["train_precision", "val_precision", "train_recall", "val_recall"],
        ["train precision", "val precision", "train recall", "val recall"],
        "Score",
        "Precision / Recall vs Epoch",
        "precision_recall_curve.png",
    )


def main() -> None:
    train_dataset = ImageDataset(metadata_file=TRAIN_MANIFEST, stats_file=TRAIN_STATS)
    val_dataset = ImageDataset(metadata_file=VAL_MANIFEST, stats_file=TRAIN_STATS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SnowCNN(in_channels=6, num_classes=2).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = make_history()
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        add_to_history(history, train_metrics, val_metrics)

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), OUTPUT_DIR / "best_cnn_segmentation_model.pt")

    torch.save(model.state_dict(), OUTPUT_DIR / "last_cnn_segmentation_model.pt")

    with open(OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_history(history)

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()