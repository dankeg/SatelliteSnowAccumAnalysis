"""ONNX Runtime helpers for the snow segmentation demo."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "model" / "best_cnn_segmentation_model.onnx"
DEFAULT_STATS_FILE = APP_DIR / "model" / "train_channel_stats.json"
DEFAULT_TARGET_SIZE = 256


def load_channel_stats(stats_file: str | Path = DEFAULT_STATS_FILE) -> tuple[np.ndarray, np.ndarray]:
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32).reshape(6, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(6, 1, 1)
    return mean, std


def load_session(model_path: str | Path = DEFAULT_MODEL) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def to_six_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.shape[-1] == 3:
        if image.max() > 1.5:
            image = image / 255.0

        image = np.moveaxis(image.astype(np.float32), -1, 0)
        gray = image.mean(axis=0, keepdims=True)
        return np.concatenate([image, gray, image[:2]], axis=0)

    image = np.moveaxis(image.astype(np.float32), -1, 0)

    if image.shape[0] < 6:
        extra = np.repeat(image[-1:], 6 - image.shape[0], axis=0)
        image = np.concatenate([image, extra], axis=0)

    return image[:6]


def prepare_input_tensor(
    raw_bytes: bytes,
    filename: str,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mean, std = load_channel_stats()

    if filename.lower().endswith(".npz"):
        data = np.load(BytesIO(raw_bytes), allow_pickle=False)
        image = np.array(data["image"], dtype=np.float32)
        source_kind = "npz"
    else:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        image = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        source_kind = "image"

    image = to_six_channels(image)

    if image.shape[1:] != (target_size, target_size):
        channels = []
        for channel in image:
            resized = Image.fromarray(channel, mode="F").resize(
                (target_size, target_size),
                Image.Resampling.BILINEAR,
            )
            channels.append(np.array(resized, dtype=np.float32))
        image = np.stack(channels, axis=0)

    visual_tensor = np.clip(image, 0.0, 1.0)[None, ...]
    input_tensor = ((image - mean) / (std + 1e-6)).astype(np.float32)[None, ...]

    meta = {
        "source_kind": source_kind,
        "target_size": target_size,
    }

    return input_tensor, visual_tensor, meta