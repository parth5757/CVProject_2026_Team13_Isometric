"""
data_loader.py
--------------
Step 1 — Load annotation CSVs, metadata, and verify image availability.
Provides a summary of dataset statistics.
"""

import os
import pandas as pd
from src.config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV, METADATA_CSV, IMAGES_DIR
)


def parse_boxes(box_string):
    """
    Convert a BoxesString like '481 820 604 922;655 957 732 1024'
    into a list of [xmin, ymin, xmax, ymax] integer lists.
    Returns an empty list when the string is empty or NaN.
    """
    if pd.isna(box_string) or str(box_string).strip() == "":
        return []
    boxes = []
    for box in str(box_string).split(";"):
        parts = box.strip().split()
        if len(parts) == 4:
            boxes.append([int(p) for p in parts])
    return boxes


def load_annotations():
    """Load train / val / test CSVs and return them as DataFrames."""
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, val_df, test_df


def load_metadata():
    """Load the metadata CSV (semicolon-separated)."""
    meta_df = pd.read_csv(METADATA_CSV, sep=";")
    return meta_df


def verify_images(annotation_dfs, images_dir=IMAGES_DIR):
    """
    Cross-check image references in annotation DataFrames against the
    actual images directory.

    Returns
    -------
    all_referenced : set  – image names referenced in CSVs
    actual_images  : set  – image files on disk
    missing_images : set  – referenced but not on disk
    orphan_images  : set  – on disk but not referenced
    """
    all_referenced = set()
    for df in annotation_dfs:
        all_referenced.update(df["image_name"].unique())

    actual_images = set(os.listdir(images_dir))
    missing_images = all_referenced - actual_images
    orphan_images = actual_images - all_referenced

    return all_referenced, actual_images, missing_images, orphan_images


def print_summary(train_df, val_df, test_df, meta_df):
    """Print a human-readable summary of the loaded dataset."""
    all_dfs = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Count total wheat heads (bounding boxes)
    total_heads = 0
    for box_str in all_dfs["BoxesString"]:
        total_heads += len(parse_boxes(box_str))

    unique_domains = all_dfs["domain"].nunique()
    unique_countries = meta_df["country"].nunique()

    print("=" * 60)
    print("  GWHD 2021 — Dataset Summary")
    print("=" * 60)
    print(f"  Total images (all splits) : {len(all_dfs)}")
    print(f"  Total wheat heads (boxes) : {total_heads}")
    print(f"  Unique domains            : {unique_domains}")
    print(f"  Countries (from metadata) : {unique_countries}")
    print("-" * 60)
    print(f"  Train split size          : {len(train_df)}")
    print(f"  Validation split size     : {len(val_df)}")
    print(f"  Test split size           : {len(test_df)}")
    print("=" * 60)

    return {
        "total_images": len(all_dfs),
        "total_heads": total_heads,
        "unique_domains": unique_domains,
        "unique_countries": unique_countries,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }


def run_data_loading():
    """
    Convenience function: load everything, verify, and print summary.
    Returns all loaded DataFrames and verification results.
    """
    print("\n[Step 1] Loading data...")

    train_df, val_df, test_df = load_annotations()
    meta_df = load_metadata()

    all_ref, actual, missing, orphan = verify_images([train_df, val_df, test_df])
    print(f"  Images referenced in CSVs : {len(all_ref)}")
    print(f"  Images found on disk      : {len(actual)}")
    print(f"  Missing from disk         : {len(missing)}")
    print(f"  Orphan (unreferenced)     : {len(orphan)}")

    summary = print_summary(train_df, val_df, test_df, meta_df)

    return train_df, val_df, test_df, meta_df, summary


if __name__ == "__main__":
    run_data_loading()
