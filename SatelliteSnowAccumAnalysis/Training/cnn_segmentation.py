from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "SatelliteSnowAccumAnalysis" / "Demo" / "model" / "best_cnn_segmentation_model.pt"
STATS_FILE = REPO_ROOT / "snow_dataset_small" / "metadata" / "train_channel_stats.json"
TARGET_SIZE = 256


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SnowCNN(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.dec3 = DoubleConv(256, 128)
        self.dec2 = DoubleConv(128, 64)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.bottleneck(self.pool(x3))

        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.out(x)


def load_channel_stats(stats_file: str | Path = STATS_FILE) -> tuple[torch.Tensor, torch.Tensor]:
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
    return mean, std


def load_model(checkpoint_path=CHECKPOINT, device=None):
    device = device or torch.device("cpu")

    model = SnowCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)

    fixed_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace(".block.", ".layers.")
        key = key.replace("out_conv.", "out.")
        fixed_state_dict[key] = value

    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model


def to_six_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]

    if image.shape[-1] in (1, 3):
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.max() > 1.5:
            image = image / 255.0

        gray = image.mean(axis=-1, keepdims=True)
        extra = np.concatenate([gray, image[..., :2]], axis=-1)
        image = np.concatenate([image, extra], axis=-1)
        return np.moveaxis(image, -1, 0)

    if image.shape[0] in (1, 3):
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        if image.max() > 1.5:
            image = image / 255.0

        gray = image.mean(axis=0, keepdims=True)
        extra = np.concatenate([gray, image[:2]], axis=0)
        return np.concatenate([image, extra], axis=0)

    return image[:6]


def prepare_input_tensor(
    raw_bytes: bytes,
    filename: str,
    target_size: int = TARGET_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    suffix = Path(filename).suffix.lower()

    if suffix == ".npz":
        image = np.load(BytesIO(raw_bytes))["image"].astype(np.float32)
        source_kind = "npz"
    else:
        pil = Image.open(BytesIO(raw_bytes)).convert("RGB")
        pil = pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
        image = np.asarray(pil, dtype=np.float32) / 255.0
        source_kind = "image"

    if image.ndim == 3 and image.shape[0] not in (1, 3, 6):
        image = np.moveaxis(image, -1, 0)

    image = to_six_channels(image).astype(np.float32)
    visual = torch.from_numpy(image).unsqueeze(0)

    image = torch.from_numpy(image)
    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    mean, std = load_channel_stats()
    image = (image - mean) / (std + 1e-6)

    meta = {
        "source_kind": source_kind,
        "target_size": target_size,
    }

    return image.unsqueeze(0), visual, meta