"""End-to-end pipeline: download -> clean -> train -> save model.

Usage:
    python scripts/run_pipeline.py              # download (if needed) and train
    python scripts/run_pipeline.py --no-download  # use data/raw/listings.csv.gz as-is
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_price.data import build_clean_dataset  # noqa: E402
from airbnb_price.train import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    df = build_clean_dataset(download=not args.no_download)
    metrics = train(df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
