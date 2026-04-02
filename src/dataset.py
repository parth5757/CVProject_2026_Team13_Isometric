"""
dataset.py
----------
Step 6 — PyTorch Dataset for the GWHD 2021 wheat-head detection task.
Loads images, applies resize + augmentation on-the-fly, and returns
tensors ready for model training.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import IMAGES_DIR, ORIGINAL_IMG_SIZE, TARGET_IMG_SIZE
from src.data_loader import parse_boxes
from src.preprocessing import (
    resize_image, rescale_boxes, normalize_image,
    get_augmentation_pipeline, apply_augmentation,
)


class WheatDataset(Dataset):
    """
    PyTorch Dataset for GWHD 2021.

    Parameters
    ----------
    df          : pd.DataFrame with columns [image_name, BoxesString, domain]
    images_dir  : path to the folder containing .png images
    target_size : resize images to this square dimension
    augment     : whether to apply data augmentation
    transform   : optional custom albumentations pipeline (overrides default)

    Returns (from __getitem__)
    --------------------------
    image  : torch.FloatTensor  (3, target_size, target_size), normalised [0,1]
    target : dict with keys
        boxes  : torch.FloatTensor (N, 4) — [x1, y1, x2, y2] in resized coords
        labels : torch.LongTensor  (N,)   — all zeros (single class: wheat head)
    """

    def __init__(self, df, images_dir=IMAGES_DIR,
                 target_size=TARGET_IMG_SIZE, augment=False, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.target_size = target_size
        self.augment = augment

        # Build augmentation pipeline (only used when augment=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_augmentation_pipeline(target_size) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────────────────────────────────
        img_path = os.path.join(self.images_dir, row["image_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ── Parse bounding boxes ──────────────────────────────────────────
        boxes = parse_boxes(row["BoxesString"])

        # ── Resize ────────────────────────────────────────────────────────
        image = resize_image(image, self.target_size)
        boxes = rescale_boxes(boxes, ORIGINAL_IMG_SIZE, self.target_size)

        # ── Augment (training only) ──────────────────────────────────────
        if self.augment and self.transform is not None and len(boxes) > 0:
            image, boxes = apply_augmentation(image, boxes, self.transform)

        # ── Normalise pixels to [0, 1] ───────────────────────────────────
        image = normalize_image(image)

        # ── Convert to tensors ────────────────────────────────────────────
        # image: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        labels_tensor = torch.zeros(len(boxes), dtype=torch.long)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image_tensor, target


# ── Custom collate function for DataLoader ────────────────────────────────────
# Detection tasks have variable numbers of boxes per image, so the default
# collate (which tries to stack) will fail.  This function keeps targets as
# a list of dicts.

def collate_fn(batch):
    """
    Custom collate for detection datasets.

    Returns
    -------
    images  : torch.FloatTensor (B, 3, H, W)
    targets : list[dict] of length B
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


if __name__ == "__main__":
    # Quick smoke test
    from src.data_loader import load_annotations

    train_df, _, _ = load_annotations()
    ds = WheatDataset(train_df, augment=True)
    img, tgt = ds[0]
    print(f"Image tensor shape : {img.shape}")
    print(f"Boxes              : {tgt['boxes'].shape}")
    print(f"Labels             : {tgt['labels'].shape}")
