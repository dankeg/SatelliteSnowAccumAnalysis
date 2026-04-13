"""Reusable UNet-like snow segmentation model and inference helpers."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "SatelliteSnowAccumAnalysis" / "Demo" / "model" / "best_cnn_segmentation_model.pt"
LEGACY_CHECKPOINT = (
    REPO_ROOT
    / "remote_run_artifacts"
    / "vastai_cnn_run_20260412_0214"
    / "cnn_training_outputs"
    / "best_cnn_segmentation_model.pt"
)
DEFAULT_STATS_FILE = REPO_ROOT / "snow_dataset_small" / "metadata" / "train_channel_stats.json"
DEFAULT_TARGET_SIZE = 256


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetLikeCNN(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.out_conv(x)


@lru_cache(maxsize=1)
def load_channel_stats(stats_file: str | Path = DEFAULT_STATS_FILE) -> tuple[torch.Tensor, torch.Tensor]:
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return mean, std


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> UNetLikeCNN:
    device = device or torch.device("cpu")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists() and LEGACY_CHECKPOINT.exists():
        checkpoint_path = LEGACY_CHECKPOINT

    model = UNetLikeCNN(in_channels=6, num_classes=2).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _expand_to_six_channels(image: np.ndarray) -> np.ndarray:
    """Convert a 2D/3D image into 6 channels for the trained segmentation CNN."""
    if image.ndim == 2:
        image = image[:, :, None]

    if image.shape[0] == 6:
        return image

    if image.shape[-1] in (1, 3):
        rgb = image.astype(np.float32)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        gray = rgb.mean(axis=-1, keepdims=True)
        derived = np.concatenate([gray, rgb[..., :2]], axis=-1)
        stacked = np.concatenate([rgb, derived], axis=-1)
        return np.moveaxis(stacked, -1, 0)

    if image.shape[0] in (1, 3):
        chw = image.astype(np.float32)
        if chw.shape[0] == 1:
            chw = np.repeat(chw, 3, axis=0)
        if chw.max() > 1.5:
            chw = chw / 255.0
        gray = chw.mean(axis=0, keepdims=True)
        derived = np.concatenate([gray, chw[:2]], axis=0)
        return np.concatenate([chw, derived], axis=0)

    if image.shape[0] < 6:
        pad = np.repeat(image[-1:, ...], 6 - image.shape[0], axis=0)
        return np.concatenate([image, pad], axis=0)

    return image[:6]


def prepare_input_tensor(
    raw_bytes: bytes,
    filename: str,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return a normalized tensor, a visualization tensor, and metadata."""
    suffix = Path(filename.lower()).suffix
    mean, std = load_channel_stats()

    if suffix == ".npz":
        payload = np.load(BytesIO(raw_bytes), allow_pickle=False)
        if "image" not in payload:
            raise ValueError("NPZ file must contain an 'image' array.")
        image = np.asarray(payload["image"], dtype=np.float32)
        source_kind = "npz"
    else:
        pil = Image.open(BytesIO(raw_bytes)).convert("RGB")
        pil = pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
        image = np.asarray(pil, dtype=np.float32) / 255.0
        source_kind = "image"

    if image.ndim == 3 and image.shape[0] not in (1, 3, 6):
        image = np.moveaxis(image, -1, 0)

    image = _expand_to_six_channels(image).astype(np.float32)
    visual_tensor = torch.from_numpy(image.copy())
    image = torch.from_numpy(image)

    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if image.shape[0] != mean.shape[0]:
        if image.shape[0] < mean.shape[0]:
            pad = torch.zeros(mean.shape[0] - image.shape[0], 1, 1, dtype=torch.float32)
            image = torch.cat([image, pad.expand(-1, image.shape[1], image.shape[2])], dim=0)
        else:
            image = image[: mean.shape[0]]

    image = (image - mean) / (std + 1e-6)
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    visual_tensor = torch.nan_to_num(visual_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    visual_tensor = torch.clamp(visual_tensor, 0.0, 1.0)

    meta = {
        "source_kind": source_kind,
        "input_channels": int(image.shape[0]),
        "input_height": int(image.shape[1]),
        "input_width": int(image.shape[2]),
        "target_size": int(target_size),
    }
    return image.unsqueeze(0), visual_tensor.unsqueeze(0), meta
