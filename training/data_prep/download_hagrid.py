"""
Download / stage the HaGRID (Hand Gesture Recognition Image Dataset) subset
used for GestureLens training.

On Kaggle, the dataset is typically added as a Kaggle Dataset input rather
than downloaded here. This script is for local/offline use.

Usage:
    python download_hagrid.py --classes call fist like ok peace --out ./data/hagrid
"""
import argparse
import os


HAGRID_INFO_URL = "https://github.com/hukenovs/hagrid"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage HaGRID dataset subset")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Gesture class names to include, e.g. call fist like ok peace",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./data/hagrid",
        help="Output directory to stage the dataset into",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[GestureLens] Staging HaGRID classes: {args.classes}")
    print(f"[GestureLens] Output directory: {args.out}")
    print(
        "[GestureLens] NOTE: HaGRID is large. Follow the official dataset "
        f"instructions at {HAGRID_INFO_URL} to download the class-specific "
        "archives, then point --out at the extracted location, or attach "
        "the Kaggle dataset directly in your training notebook."
    )

    # [TODO] Implement actual download/extraction logic, e.g.:
    #   - kagglehub.dataset_download(...)
    #   - or requests + zipfile extraction per class archive


if __name__ == "__main__":
    main()
