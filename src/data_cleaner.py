"""
data_cleaner.py
---------------
Step 2 — Validate and clean annotation CSVs.
Checks for missing values, invalid/out-of-bounds bounding boxes,
duplicates, and image availability.  Saves cleaned CSVs.
"""

import os
import pandas as pd
from tqdm import tqdm

from src.config import IMAGES_DIR, PROCESSED_DIR, ORIGINAL_IMG_SIZE
from src.data_loader import parse_boxes


# ── Helper: rebuild BoxesString from a list of boxes ─────────────────────────
def _boxes_to_string(boxes):
    """Convert list of [x1, y1, x2, y2] back to 'x1 y1 x2 y2;...' string."""
    if not boxes:
        return ""
    return ";".join(f"{b[0]} {b[1]} {b[2]} {b[3]}" for b in boxes)


# ── Core validation on a single DataFrame ────────────────────────────────────
def validate_and_clean(df, images_dir=IMAGES_DIR, img_size=ORIGINAL_IMG_SIZE):
    """
    Validate a single annotation DataFrame.

    Returns
    -------
    clean_df : pd.DataFrame  – cleaned annotations
    report   : dict           – counts of each issue found
    """
    report = {
        "missing_values_removed": 0,
        "invalid_boxes_removed": 0,
        "oob_boxes_clipped": 0,
        "duplicate_rows_removed": 0,
        "missing_image_rows_removed": 0,
        "total_rows_before": len(df),
        "images_affected": set(),
    }

    # 1. Missing values — drop rows with NaN image_name or domain
    mask_missing = df["image_name"].isna() | df["domain"].isna()
    report["missing_values_removed"] = int(mask_missing.sum())
    df = df[~mask_missing].copy()

    # 2. Duplicate full-row removal
    dup_mask = df.duplicated(keep="first")
    report["duplicate_rows_removed"] = int(dup_mask.sum())
    if dup_mask.any():
        report["images_affected"].update(df.loc[dup_mask, "image_name"].tolist())
    df = df[~dup_mask].copy()

    # 3. Check images exist on disk
    actual_images = set(os.listdir(images_dir))
    missing_mask = ~df["image_name"].isin(actual_images)
    report["missing_image_rows_removed"] = int(missing_mask.sum())
    if missing_mask.any():
        report["images_affected"].update(df.loc[missing_mask, "image_name"].tolist())
    df = df[~missing_mask].copy()

    # 4. Validate and clean bounding boxes row by row
    cleaned_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating boxes"):
        boxes = parse_boxes(row["BoxesString"])
        valid_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            w = x2 - x1
            h = y2 - y1

            # Skip boxes with non-positive width or height
            if w <= 0 or h <= 0:
                report["invalid_boxes_removed"] += 1
                report["images_affected"].add(row["image_name"])
                continue

            # Clip boxes to image boundaries
            clipped = False
            if x1 < 0:
                x1, clipped = 0, True
            if y1 < 0:
                y1, clipped = 0, True
            if x2 > img_size:
                x2, clipped = img_size, True
            if y2 > img_size:
                y2, clipped = img_size, True
            if clipped:
                report["oob_boxes_clipped"] += 1

            valid_boxes.append([x1, y1, x2, y2])

        new_row = row.copy()
        new_row["BoxesString"] = _boxes_to_string(valid_boxes)
        cleaned_rows.append(new_row)

    clean_df = pd.DataFrame(cleaned_rows)
    clean_df.reset_index(drop=True, inplace=True)

    report["total_rows_after"] = len(clean_df)
    report["rows_removed"] = report["total_rows_before"] - report["total_rows_after"]
    report["images_affected"] = len(report["images_affected"])

    return clean_df, report


# ── Public entry point ────────────────────────────────────────────────────────
def run_data_cleaning(train_df, val_df, test_df):
    """
    Clean all three splits. Save results to processed_data/ and print reports.
    """
    print("\n[Step 2] Cleaning and validating annotations...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    cleaned = {}

    for name, df in splits.items():
        print(f"\n  --- {name} split ---")
        clean_df, report = validate_and_clean(df)

        out_path = os.path.join(PROCESSED_DIR, f"{name}_clean.csv")
        clean_df.to_csv(out_path, index=False)
        print(f"  Rows before          : {report['total_rows_before']}")
        print(f"  Rows after           : {report['total_rows_after']}")
        print(f"  Rows removed         : {report['rows_removed']}")
        print(f"    Missing values     : {report['missing_values_removed']}")
        print(f"    Duplicates         : {report['duplicate_rows_removed']}")
        print(f"    Missing images     : {report['missing_image_rows_removed']}")
        print(f"  Invalid boxes removed: {report['invalid_boxes_removed']}")
        print(f"  OOB boxes clipped    : {report['oob_boxes_clipped']}")
        print(f"  Images affected      : {report['images_affected']}")
        print(f"  Saved -> {out_path}")

        cleaned[name] = clean_df

    return cleaned["train"], cleaned["val"], cleaned["test"]


if __name__ == "__main__":
    from src.data_loader import load_annotations
    train_df, val_df, test_df = load_annotations()
    run_data_cleaning(train_df, val_df, test_df)
