"""Sets up an initial CNN for snow segmentation."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SatelliteSnowAccumAnalysis.Training.ImageDataset import ImageDataset


IGNORE_LABEL = 255


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetLikeCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        x4 = self.bottleneck(self.pool3(x3))

        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        x = self.out_conv(x)
        return x


def update_binary_counts(preds, labels, counts):
    valid = labels != IGNORE_LABEL
    if valid.sum().item() == 0:
        return

    preds = preds[valid]
    labels = labels[valid]

    counts["tp"] += ((preds == 1) & (labels == 1)).sum().item()
    counts["tn"] += ((preds == 0) & (labels == 0)).sum().item()
    counts["fp"] += ((preds == 1) & (labels == 0)).sum().item()
    counts["fn"] += ((preds == 0) & (labels == 1)).sum().item()


def compute_metrics(counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    num_batches = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)

            valid_pixels = (labels != IGNORE_LABEL).sum()
            if valid_pixels.item() == 0:
                continue

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if not torch.isfinite(loss):
                split_name = "training" if training else "validation"
                print(f"[warn] skipping non-finite {split_name} loss batch")
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1)
            update_binary_counts(preds, labels, counts)

    mean_loss = total_loss / num_batches if num_batches > 0 else float("nan")
    metrics = compute_metrics(counts)
    metrics["loss"] = mean_loss
    metrics["batches"] = num_batches

    return metrics


def plot_history(history, outdir):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="train accuracy")
    plt.plot(epochs, history["val_accuracy"], label="val accuracy")
    plt.plot(epochs, history["train_f1"], label="train f1")
    plt.plot(epochs, history["val_f1"], label="val f1")
    plt.plot(epochs, history["train_iou"], label="train iou")
    plt.plot(epochs, history["val_iou"], label="val iou")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Segmentation Metrics vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "metrics_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_precision"], label="train precision")
    plt.plot(epochs, history["val_precision"], label="val precision")
    plt.plot(epochs, history["train_recall"], label="train recall")
    plt.plot(epochs, history["val_recall"], label="val recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "precision_recall_curve.png")
    plt.close()


train_dataset = ImageDataset(
    metadata_file="snow_dataset_small/manifests/train.csv",
    stats_file="snow_dataset_small/metadata/train_channel_stats.json",
)

val_dataset = ImageDataset(
    metadata_file="snow_dataset_small/manifests/val.csv",
    stats_file="snow_dataset_small/metadata/train_channel_stats.json",
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetLikeCNN(in_channels=6, num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
outdir = Path("cnn_training_outputs")
outdir.mkdir(parents=True, exist_ok=True)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_precision": [],
    "val_precision": [],
    "train_recall": [],
    "val_recall": [],
    "train_f1": [],
    "val_f1": [],
    "train_iou": [],
    "val_iou": [],
}

best_val_f1 = -1.0

for epoch in range(num_epochs):
    train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
    val_metrics = run_epoch(model, val_loader, loss_fn, device, optimizer=None)

    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_accuracy"].append(train_metrics["accuracy"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["train_precision"].append(train_metrics["precision"])
    history["val_precision"].append(val_metrics["precision"])
    history["train_recall"].append(train_metrics["recall"])
    history["val_recall"].append(val_metrics["recall"])
    history["train_f1"].append(train_metrics["f1"])
    history["val_f1"].append(val_metrics["f1"])
    history["train_iou"].append(train_metrics["iou"])
    history["val_iou"].append(val_metrics["iou"])

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
        torch.save(model.state_dict(), outdir / "best_cnn_segmentation_model.pt")

torch.save(model.state_dict(), outdir / "last_cnn_segmentation_model.pt")

with open(outdir / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

plot_history(history, outdir)

print(f"\nSaved outputs to: {outdir.resolve()}")
print("Saved files:")
print(f"  - {outdir / 'best_cnn_segmentation_model.pt'}")
print(f"  - {outdir / 'last_cnn_segmentation_model.pt'}")
print(f"  - {outdir / 'training_history.json'}")
print(f"  - {outdir / 'loss_curve.png'}")
print(f"  - {outdir / 'metrics_curve.png'}")
print(f"  - {outdir / 'precision_recall_curve.png'}")