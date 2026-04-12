"""Dataset utilities for Sentinel snow segmentation."""

from pathlib import Path
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        metadata_file,
        image_h=None,
        image_w=None,
        label_map=None,
        samples_root=None,
        include_ndsi=False,
        stats_file=None,
        return_meta=False,
    ):
        self.samples = []
        self.image_h = image_h
        self.image_w = image_w
        self.include_ndsi = include_ndsi
        self.return_meta = return_meta

        self.metadata_file = Path(metadata_file)

        if samples_root is None:
            # assumes metadata_file looks like: <outdir>/manifests/train.csv
            # and samples live at: <outdir>/samples/...
            self.samples_root = self.metadata_file.parent.parent / "samples"
        else:
            self.samples_root = Path(samples_root)

        self.mean = None
        self.std = None
        if stats_file is not None:
            with open(stats_file, "r") as f:
                stats = json.load(f)

            if stats.get("mean") is not None and stats.get("std") is not None:
                mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
                std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

                # just in case the stats file itself has bad values
                self.mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
                self.std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)

        with open(self.metadata_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        sample_path = self.samples_root / row["sample_relpath"]

        payload = np.load(sample_path)

        image = torch.from_numpy(payload["image"]).float()   # [C, H, W]
        label = torch.from_numpy(payload["label"]).long()    # [H, W]

        # remove bad numeric values before they can poison the model
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.include_ndsi and "ndsi" in payload:
            ndsi = torch.from_numpy(payload["ndsi"]).float().unsqueeze(0)  # [1, H, W]
            ndsi = torch.nan_to_num(ndsi, nan=0.0, posinf=0.0, neginf=0.0)
            image = torch.cat([image, ndsi], dim=0)

        if self.image_h is not None and self.image_w is not None:
            if image.shape[1] != self.image_h or image.shape[2] != self.image_w:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.image_h, self.image_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                label = F.interpolate(
                    label.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.image_h, self.image_w),
                    mode="nearest",
                ).squeeze(0).squeeze(0).long()

        if self.mean is not None and self.std is not None:
            mean = self.mean
            std = self.std

            if self.include_ndsi and image.shape[0] == mean.shape[0] + 1:
                ndsi_mean = torch.zeros(1, 1, 1, dtype=torch.float32)
                ndsi_std = torch.ones(1, 1, 1, dtype=torch.float32)
                mean = torch.cat([mean, ndsi_mean], dim=0)
                std = torch.cat([std, ndsi_std], dim=0)

            image = (image - mean) / (std + 1e-6)

        # do this again after normalization in case stats were bad
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        if self.return_meta:
            meta = {
                "sample_id": row.get("sample_id"),
                "city": row.get("city"),
                "window_year": row.get("window_year"),
                "date": row.get("date"),
                "item_id": row.get("item_id"),
                "split": row.get("split"),
            }
            return image, label, meta

        return image, label