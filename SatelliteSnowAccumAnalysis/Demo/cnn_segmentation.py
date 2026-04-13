"""Demo-local ONNX Runtime helpers for snow segmentation."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json

import numpy as np
from PIL import Image
import onnxruntime as ort


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "model" / "best_cnn_segmentation_model.onnx"
DEFAULT_STATS_FILE = APP_DIR / "model" / "train_channel_stats.json"
LEGACY_STATS_FILE = Path(__file__).resolve().parents[2] / "snow_dataset_small" / "metadata" / "train_channel_stats.json"
DEFAULT_TARGET_SIZE = 256


def resolve_stats_file(stats_file: str | Path = DEFAULT_STATS_FILE) -> Path:
    stats_path = Path(stats_file)
    if stats_path.exists():
        return stats_path

    if stats_path == DEFAULT_STATS_FILE and LEGACY_STATS_FILE.exists():
        return LEGACY_STATS_FILE

    raise FileNotFoundError(
        "Channel stats file not found. Expected "
        f"{stats_path}"
        + (f" or legacy path {LEGACY_STATS_FILE}" if stats_path == DEFAULT_STATS_FILE else "")
    )


@lru_cache(maxsize=1)
def load_channel_stats(stats_file: str | Path = DEFAULT_STATS_FILE) -> tuple[np.ndarray, np.ndarray]:
    with open(resolve_stats_file(stats_file), "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    return mean, std


@lru_cache(maxsize=1)
def load_session(model_path: str | Path = DEFAULT_MODEL) -> ort.InferenceSession:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _expand_to_six_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]

    if image.shape[-1] == 6:
        chw = np.moveaxis(image, -1, 0)
        return chw.astype(np.float32)

    if image.shape[-1] in (1, 3):
        rgb = image.astype(np.float32)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        chw = np.moveaxis(rgb, -1, 0)
        gray = chw.mean(axis=0, keepdims=True)
        derived = np.concatenate([gray, chw[:2]], axis=0)
        return np.concatenate([chw, derived], axis=0)

    chw = np.moveaxis(image, -1, 0).astype(np.float32)
    if chw.shape[0] < 6:
        pad = np.repeat(chw[-1:, ...], 6 - chw.shape[0], axis=0)
        chw = np.concatenate([chw, pad], axis=0)
    return chw[:6]


def prepare_input_tensor(
    raw_bytes: bytes,
    filename: str,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray, dict]:
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

    image = _expand_to_six_channels(image)

    if image.shape[1:] != (target_size, target_size):
        resized = []
        for channel in image:
            resized.append(
                np.asarray(
                    Image.fromarray(channel.astype(np.float32), mode="F").resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
            )
        image = np.stack(resized, axis=0)

    visual_tensor = np.clip(image.copy(), 0.0, 1.0).astype(np.float32)
    image = (image.astype(np.float32) - mean) / (std + 1e-6)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    meta = {
        "source_kind": source_kind,
        "input_channels": int(image.shape[0]),
        "input_height": int(image.shape[1]),
        "input_width": int(image.shape[2]),
        "target_size": int(target_size),
    }
    return image[np.newaxis, ...], visual_tensor[np.newaxis, ...], meta
