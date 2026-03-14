"""Helper Class to Abstract Out Image Dataset Operations"""

import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, metadata_file, image_h, image_w, label_map):
        self.data = []
        self.transform = transforms.Compose([
                    transforms.Resize((image_w, image_h)),
                    transforms.ToTensor(),
                ])
        
        with open(metadata_file, "r") as data_file:
            for line in data_file:
                item = json.loads(line)
                self.data.append((item["image_path"], label_map[item["label"]]))

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            image_path, label = self.samples[idx]
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image, label 