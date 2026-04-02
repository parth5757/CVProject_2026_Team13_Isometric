"""
preprocessing.py
----------------
Step 4 — Data Preprocessing for model training.
  - Image resizing (1024 -> 640) with coordinate rescaling
  - Pixel normalisation to [0, 1]
  - Albumentations augmentation pipeline
  - YOLO-format label conversion and saving
"""

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations import BboxParams
from tqdm import tqdm

from src.config import (
    IMAGES_DIR, PROCESSED_DIR, YOLO_LABELS_DIR,
    ORIGINAL_IMG_SIZE, TARGET_IMG_SIZE, AUGMENTATION_SEED
)
from src.data_loader import parse_boxes


# ═════════════════════════════════════════════════════════════════════════════
# 1. Resize helpers
# ═════════════════════════════════════════════════════════════════════════════

def resize_image(image, target_size=TARGET_IMG_SIZE):
    """Resize an image to (target_size, target_size) using bilinear interpolation."""
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def rescale_boxes(boxes, orig_size=ORIGINAL_IMG_SIZE, target_size=TARGET_IMG_SIZE):
    """
    Scale bounding boxes from orig_size to target_size.
    boxes: list of [x1, y1, x2, y2]
    """
    scale = target_size / orig_size
    return [[int(b[0] * scale), int(b[1] * scale),
             int(b[2] * scale), int(b[3] * scale)] for b in boxes]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Normalisation
# ═════════════════════════════════════════════════════════════════════════════

def normalize_image(image):
    """Scale pixel values from [0, 255] to [0.0, 1.0] float32."""
    return image.astype(np.float32) / 255.0


# ═════════════════════════════════════════════════════════════════════════════
# 3. Augmentation
# ═════════════════════════════════════════════════════════════════════════════

def get_augmentation_pipeline(target_size=TARGET_IMG_SIZE):
    """
    Return an Albumentations Compose pipeline with bbox-safe transforms.
    Bounding boxes use 'pascal_voc' format: [x_min, y_min, x_max, y_max].
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                 scale=(0.9, 1.1), rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ], bbox_params=BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.3,        # drop boxes that become mostly invisible
    ))


def apply_augmentation(image, boxes, transform=None):
    """
    Apply augmentation to an image and its bounding boxes.

    Parameters
    ----------
    image  : np.ndarray (H, W, 3) uint8  – already resized
    boxes  : list of [x1, y1, x2, y2]
    transform : albumentations.Compose or None

    Returns
    -------
    aug_image : np.ndarray
    aug_boxes : list of [x1, y1, x2, y2]
    """
    if transform is None:
        transform = get_augmentation_pipeline()

    # All boxes belong to a single class (wheat head -> 0)
    class_labels = [0] * len(boxes)

    if len(boxes) == 0:
        transformed = transform(image=image, bboxes=[], class_labels=[])
        return transformed["image"], []

    transformed = transform(image=image, bboxes=boxes, class_labels=class_labels)
    aug_boxes = [list(b) for b in transformed["bboxes"]]
    return transformed["image"], aug_boxes


# ═════════════════════════════════════════════════════════════════════════════
# 4. YOLO format conversion
# ═════════════════════════════════════════════════════════════════════════════

def boxes_to_yolo(boxes, img_size=TARGET_IMG_SIZE):
    """
    Convert [x1, y1, x2, y2] boxes to YOLO format:
      class_id  x_center  y_center  width  height
    All values normalised by img_size. class_id = 0 (wheat head).
    """
    yolo_labels = []
    for b in boxes:
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0 / img_size
        cy = (y1 + y2) / 2.0 / img_size
        nw = w / img_size
        nh = h / img_size
        yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return yolo_labels


def save_yolo_labels(df, save_dir=YOLO_LABELS_DIR,
                     orig_size=ORIGINAL_IMG_SIZE, target_size=TARGET_IMG_SIZE):
    """
    For every row in df, convert boxes to YOLO format and save a .txt file.
    The label files are named <image_stem>.txt.
    """
    os.makedirs(save_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing YOLO labels"):
        boxes = parse_boxes(row["BoxesString"])
        boxes = rescale_boxes(boxes, orig_size, target_size)
        yolo_lines = boxes_to_yolo(boxes, img_size=target_size)
        stem = os.path.splitext(row["image_name"])[0]
        out_path = os.path.join(save_dir, f"{stem}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_preprocessing(train_df, val_df, test_df):
    """
    Run full preprocessing: YOLO label generation for all splits.
    Image resizing / augmentation happen on-the-fly via the Dataset class
    (Step 6), so here we persist only the label files.
    """
    print("\n[Step 4] Data Preprocessing...")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"  Converting {len(all_df)} annotation rows -> YOLO labels...")
    save_yolo_labels(all_df)
    print(f"  YOLO labels saved -> {YOLO_LABELS_DIR}")

    # Quick demo: resize + augment one image to verify the pipeline works
    sample_row = train_df.iloc[0]
    img_path = os.path.join(IMAGES_DIR, sample_row["image_name"])
    img = cv2.imread(img_path)
    if img is not None:
        boxes = parse_boxes(sample_row["BoxesString"])
        img_r = resize_image(img)
        boxes_r = rescale_boxes(boxes)
        transform = get_augmentation_pipeline()
        aug_img, aug_boxes = apply_augmentation(img_r, boxes_r, transform)
        print(f"  Demo: resized to {img_r.shape[:2]}, "
              f"augmented boxes {len(boxes_r)} -> {len(aug_boxes)}")
    print("  [Preprocessing complete]")


if __name__ == "__main__":
    from src.data_loader import load_annotations
    train_df, val_df, test_df = load_annotations()
    run_preprocessing(train_df, val_df, test_df)
