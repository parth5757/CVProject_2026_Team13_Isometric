"""
pipeline.py
-----------
Step 5 — Reusable preprocessing pipeline class.
Orchestrates every step (load -> clean -> EDA -> preprocess) in one call.
"""

import os
import pandas as pd

from src.config import PROCESSED_DIR
from src.data_loader import (
    load_annotations, load_metadata, verify_images,
    print_summary, run_data_loading,
)
from src.data_cleaner import run_data_cleaning
from src.eda import run_eda
from src.preprocessing import (
    resize_image, rescale_boxes, normalize_image,
    get_augmentation_pipeline, apply_augmentation,
    save_yolo_labels, run_preprocessing,
)


class WheatDatasetPreprocessor:
    """
    End-to-end preprocessing pipeline for the GWHD 2021 dataset.

    Usage
    -----
        pipeline = WheatDatasetPreprocessor()
        pipeline.run()          # run everything sequentially
        # — or call individual steps —
        pipeline.load_annotations()
        pipeline.validate_annotations()
        pipeline.perform_eda()
        pipeline.resize_images()           # on-the-fly via Dataset class
        pipeline.apply_augmentations()     # on-the-fly via Dataset class
        pipeline.convert_to_yolo_format()
    """

    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.meta_df = None
        self.summary = None
        # Cleaned versions (populated after validate_annotations)
        self.train_clean = None
        self.val_clean = None
        self.test_clean = None

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    def load_annotations(self):
        """Load raw CSVs and metadata; print dataset summary."""
        self.train_df, self.val_df, self.test_df, self.meta_df, self.summary = (
            run_data_loading()
        )
        return self

    # ── Step 2: Validate & Clean ─────────────────────────────────────────────
    def validate_annotations(self):
        """Clean annotations and save to processed_data/."""
        if self.train_df is None:
            self.load_annotations()
        self.train_clean, self.val_clean, self.test_clean = run_data_cleaning(
            self.train_df, self.val_df, self.test_df
        )
        return self

    # ── Step 3: EDA ──────────────────────────────────────────────────────────
    def perform_eda(self):
        """Generate all EDA plots from the cleaned data."""
        # Use cleaned data if available, otherwise raw
        t = self.train_clean if self.train_clean is not None else self.train_df
        v = self.val_clean if self.val_clean is not None else self.val_df
        te = self.test_clean if self.test_clean is not None else self.test_df
        if t is None:
            self.load_annotations()
            t, v, te = self.train_df, self.val_df, self.test_df
        run_eda(t, v, te, self.meta_df)
        return self

    # ── Step 4a: Resize (docs) ───────────────────────────────────────────────
    def resize_images(self):
        """
        Image resizing is handled on-the-fly by the WheatDataset class
        to avoid duplicating the full image set on disk.
        This method exists for API completeness and prints guidance.
        """
        print("\n[Step 4a] Image resizing")
        print("  Resizing is applied on-the-fly inside WheatDataset.__getitem__().")
        print("  Use preprocessing.resize_image() for batch offline resizing.")
        return self

    # ── Step 4b: Augment (docs) ──────────────────────────────────────────────
    def apply_augmentations(self):
        """
        Augmentations are applied on-the-fly inside WheatDataset.__getitem__().
        This method exists for API completeness.
        """
        print("\n[Step 4b] Augmentations")
        print("  Augmentations are applied on-the-fly inside WheatDataset.__getitem__().")
        print("  Use preprocessing.get_augmentation_pipeline() for standalone use.")
        return self

    # ── Step 4c: YOLO conversion ─────────────────────────────────────────────
    def convert_to_yolo_format(self):
        """Convert annotations -> YOLO .txt labels."""
        t = self.train_clean if self.train_clean is not None else self.train_df
        v = self.val_clean if self.val_clean is not None else self.val_df
        te = self.test_clean if self.test_clean is not None else self.test_df
        if t is None:
            self.load_annotations()
            t, v, te = self.train_df, self.val_df, self.test_df
        run_preprocessing(t, v, te)
        return self

    # ── Run all ──────────────────────────────────────────────────────────────
    def run(self):
        """Execute the full pipeline end-to-end."""
        print("=" * 60)
        print("  GWHD 2021 — Full Preprocessing Pipeline")
        print("=" * 60)
        self.load_annotations()
        self.validate_annotations()
        self.perform_eda()
        self.resize_images()
        self.apply_augmentations()
        self.convert_to_yolo_format()
        print("\n" + "=" * 60)
        print("  Pipeline complete.")
        print("=" * 60)
        return self


if __name__ == "__main__":
    pipeline = WheatDatasetPreprocessor()
    pipeline.run()
