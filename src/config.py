"""
Centralized configuration for the GWHD 2021 preprocessing pipeline.
All paths and hyperparameters are defined here for easy modification.
"""

import os

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Raw data paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "gwhd_2021")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TRAIN_CSV = os.path.join(DATA_DIR, "competition_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "competition_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "competition_test.csv")
METADATA_CSV = os.path.join(DATA_DIR, "metadata_dataset.csv")

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_PLOTS_DIR = os.path.join(OUTPUT_DIR, "eda_plots")
SAMPLE_ANNOTATIONS_DIR = os.path.join(EDA_PLOTS_DIR, "sample_annotations")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
YOLO_LABELS_DIR = os.path.join(PROCESSED_DIR, "yolo_labels")

# ── Image parameters ─────────────────────────────────────────────────────────
ORIGINAL_IMG_SIZE = 1024   # Original image dimension (square)
TARGET_IMG_SIZE = 640      # Resized image dimension (square)

# ── Augmentation parameters ──────────────────────────────────────────────────
AUGMENTATION_SEED = 42
