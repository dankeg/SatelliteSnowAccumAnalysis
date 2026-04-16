"""Export the base image and the two class logit CSVs for each saved snow sample."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from SatelliteSnowAccumAnalysis.Training.cnn_segmentation import (
    REPO_ROOT,
    load_channel_stats,
    load_model,
)

INPUT_DIR = REPO_ROOT / "snow_dataset_small" / "samples"
OUTPUT_DIR = REPO_ROOT / "csv_prediction_maps"
CHECKPOINT = REPO_ROOT / "cnn_training_outputs" / "best_cnn_segmentation_model.pt"


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device=device)
    mean, std = load_channel_stats()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(INPUT_DIR.rglob("*.npz")):
        print(f"processing {file_path.name}")

        with np.load(file_path) as data:
            image = data["image"].astype(np.float32)

        x = torch.from_numpy(image).float()
        x = (x - mean) / (std + 1e-6)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x).squeeze(0).cpu().numpy()

        sample_dir = OUTPUT_DIR / file_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.moveaxis(image[[2, 1, 0]], 0, -1)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)

        save_image(sample_dir / "image.png", rgb)
        np.savetxt(sample_dir / "class0.csv", logits[0], fmt="%.6f", delimiter=",")
        np.savetxt(sample_dir / "class1.csv", logits[1], fmt="%.6f", delimiter=",")

        print(f"saved {sample_dir}")


if __name__ == "__main__":
    main()