"""
eda.py
------
Step 3 — Exploratory Data Analysis.
Generates statistics and plots for the GWHD 2021 dataset.
All plots are saved to outputs/eda_plots/.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.config import (
    IMAGES_DIR, EDA_PLOTS_DIR, SAMPLE_ANNOTATIONS_DIR, ORIGINAL_IMG_SIZE
)
from src.data_loader import parse_boxes


# ── Utility: extract per-box stats from a DataFrame ──────────────────────────
def _extract_box_stats(df):
    """Return lists of widths, heights, areas, and per-image head counts."""
    widths, heights, areas = [], [], []
    heads_per_image = []

    for box_str in df["BoxesString"]:
        boxes = parse_boxes(box_str)
        heads_per_image.append(len(boxes))
        for b in boxes:
            w = b[2] - b[0]
            h = b[3] - b[1]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    return widths, heights, areas, heads_per_image


# ── 1. Dataset overview ──────────────────────────────────────────────────────
def dataset_overview(df):
    widths, heights, areas, heads = _extract_box_stats(df)
    total_images = len(df)
    total_heads = sum(heads)
    avg_heads = total_heads / total_images if total_images else 0

    print("\n  [EDA] Dataset overview")
    print(f"    Total images      : {total_images}")
    print(f"    Total wheat heads : {total_heads}")
    print(f"    Avg heads / image : {avg_heads:.2f}")

    return widths, heights, areas, heads


# ── 2. Bounding box statistics ───────────────────────────────────────────────
def plot_bbox_statistics(widths, heights, areas, save_dir=EDA_PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(widths, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title("Bounding Box Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=50, color="salmon", edgecolor="black")
    axes[1].set_title("Bounding Box Height Distribution")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Count")

    axes[2].hist(areas, bins=50, color="mediumseagreen", edgecolor="black")
    axes[2].set_title("Bounding Box Area Distribution")
    axes[2].set_xlabel("Area (px²)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(save_dir, "bbox_statistics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")


# ── 3. Wheat head density ────────────────────────────────────────────────────
def plot_heads_per_image(heads_per_image, save_dir=EDA_PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(heads_per_image, bins=40, color="darkorange", edgecolor="black")
    plt.title("Wheat Heads per Image")
    plt.xlabel("Number of Heads")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    path = os.path.join(save_dir, "heads_per_image.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")


# ── 4. Domain distribution ──────────────────────────────────────────────────
def plot_domain_distribution(df, meta_df, save_dir=EDA_PLOTS_DIR):
    """
    Bar chart: number of images per country (aggregated from domain -> country).
    Also plots images per domain for top-15 domains.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Merge annotation domains with metadata to get country info
    domain_counts = df["domain"].value_counts().reset_index()
    domain_counts.columns = ["domain", "image_count"]
    merged = domain_counts.merge(meta_df[["name", "country"]], left_on="domain",
                                  right_on="name", how="left")

    # ── Plot A: images per country ───
    country_counts = merged.groupby("country")["image_count"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    country_counts.plot(kind="bar", color="teal", edgecolor="black")
    plt.title("Number of Images per Country")
    plt.xlabel("Country")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "images_per_country.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")

    # ── Plot B: top-15 domains ───
    top15 = domain_counts.head(15)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=top15, x="domain", y="image_count", hue="domain",
                palette="viridis", legend=False)
    plt.title("Top-15 Domains by Image Count")
    plt.xlabel("Domain")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "top15_domains.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")


# ── 5. Development stage distribution ────────────────────────────────────────
def plot_development_stage(df, meta_df, save_dir=EDA_PLOTS_DIR):
    """Bar chart of images per development stage."""
    os.makedirs(save_dir, exist_ok=True)

    domain_counts = df["domain"].value_counts().reset_index()
    domain_counts.columns = ["domain", "image_count"]
    merged = domain_counts.merge(meta_df[["name", "development_stage"]],
                                  left_on="domain", right_on="name", how="left")

    # Normalize stage names to lowercase for consistent grouping
    merged["stage_clean"] = merged["development_stage"].str.strip().str.lower()
    stage_counts = merged.groupby("stage_clean")["image_count"].sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    stage_counts.plot(kind="bar", color="mediumpurple", edgecolor="black")
    plt.title("Images by Development Stage")
    plt.xlabel("Development Stage")
    plt.ylabel("Image Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "development_stage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")


# ── 6. Bounding box spatial heatmap ──────────────────────────────────────────
def plot_bbox_heatmap(df, save_dir=EDA_PLOTS_DIR, img_size=ORIGINAL_IMG_SIZE):
    """Heatmap of bounding-box centers across all images."""
    os.makedirs(save_dir, exist_ok=True)
    heatmap = np.zeros((img_size, img_size), dtype=np.float64)

    for box_str in tqdm(df["BoxesString"], desc="Building heatmap"):
        for b in parse_boxes(box_str):
            cx = (b[0] + b[2]) // 2
            cy = (b[1] + b[3]) // 2
            cx = min(cx, img_size - 1)
            cy = min(cy, img_size - 1)
            heatmap[cy, cx] += 1

    # Gaussian blur for smoother visualisation
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Density")
    plt.title("Bounding Box Center Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    path = os.path.join(save_dir, "bbox_center_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved -> {path}")


# ── 7. Visual inspection — sample annotated images ──────────────────────────
def plot_sample_annotations(df, n=20, images_dir=IMAGES_DIR,
                            save_dir=SAMPLE_ANNOTATIONS_DIR, seed=42):
    """Draw bounding boxes on n random images and save them."""
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)

    # Filter to rows that actually have boxes
    df_with_boxes = df[df["BoxesString"].apply(lambda s: len(parse_boxes(s)) > 0)]
    sample = df_with_boxes.sample(n=min(n, len(df_with_boxes)), random_state=seed)

    for _, row in sample.iterrows():
        img_path = os.path.join(images_dir, row["image_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        boxes = parse_boxes(row["BoxesString"])
        for b in boxes:
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

        out_path = os.path.join(save_dir, row["image_name"])
        cv2.imwrite(out_path, img)

    print(f"    Saved {min(n, len(df_with_boxes))} annotated samples -> {save_dir}")


# ── Public entry point ────────────────────────────────────────────────────────
def run_eda(train_df, val_df, test_df, meta_df):
    """Run all EDA analyses on the combined dataset."""
    print("\n[Step 3] Exploratory Data Analysis...")

    # Combine all splits for overall analysis
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    widths, heights, areas, heads = dataset_overview(all_df)

    print("  Generating bounding box statistics plots...")
    plot_bbox_statistics(widths, heights, areas)

    print("  Generating heads-per-image histogram...")
    plot_heads_per_image(heads)

    print("  Generating domain distribution plots...")
    plot_domain_distribution(all_df, meta_df)

    print("  Generating development stage plot...")
    plot_development_stage(all_df, meta_df)

    print("  Generating bounding box heatmap...")
    plot_bbox_heatmap(all_df)

    print("  Generating sample annotations...")
    plot_sample_annotations(all_df)

    print("  [EDA complete]")


if __name__ == "__main__":
    from src.data_loader import load_annotations, load_metadata
    train_df, val_df, test_df = load_annotations()
    meta_df = load_metadata()
    run_eda(train_df, val_df, test_df, meta_df)
