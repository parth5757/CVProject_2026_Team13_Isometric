"""
main.py
-------
Entry point for the GWHD 2021 preprocessing pipeline.
Run this script to execute every step in sequence:
  1. Data Loading
  2. Data Cleaning & Validation
  3. Exploratory Data Analysis
  4. Preprocessing (YOLO label conversion)

Usage:
    python main.py            # run the full pipeline
    python main.py --step 1   # run only Step 1
    python main.py --step 2   # run only Step 2  (requires Step 1)
    python main.py --step 3   # run only Step 3  (requires Step 1)
    python main.py --step 4   # run only Step 4  (requires Step 1)
"""

import argparse
from src.pipeline import WheatDatasetPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="GWHD 2021 Preprocessing Pipeline"
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Run a specific step (1-4). 0 = run all steps (default)."
    )
    args = parser.parse_args()

    pipeline = WheatDatasetPreprocessor()

    if args.step == 0:
        pipeline.run()
    elif args.step == 1:
        pipeline.load_annotations()
    elif args.step == 2:
        pipeline.load_annotations()
        pipeline.validate_annotations()
    elif args.step == 3:
        pipeline.load_annotations()
        pipeline.validate_annotations()
        pipeline.perform_eda()
    elif args.step == 4:
        pipeline.load_annotations()
        pipeline.validate_annotations()
        pipeline.convert_to_yolo_format()
    else:
        print(f"Unknown step: {args.step}. Use 0 (all), 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
