"""Fetch the demo datasets into ``data/raw/`` (or a custom output directory).

Standalone by design — no ml_pipeline imports — so it works before the package
is installed:

    python scripts/download_data.py --dataset titanic
    python scripts/download_data.py --dataset california
    python scripts/download_data.py --dataset all --out data/raw
"""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

TITANIC_LOCAL_SOURCE = Path(
    "/home/aryan/Desktop/ai-engineer-roadmap/ml-journey/data/raw/titanic.csv"
)
TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
TITANIC_FILENAME = "titanic.csv"
CALIFORNIA_FILENAME = "california_housing.csv"


def _log_csv_stats(path: Path) -> None:
    """Log rows/columns of a written CSV (rows exclude the header line)."""
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        n_cols = len(header.rstrip("\n").split(",")) if header else 0
        n_rows = sum(1 for _ in fh)
    logger.info("Wrote %s: %d rows x %d columns", path, n_rows, n_cols)


def download_titanic(out_dir: Path) -> Path:
    """Place titanic.csv into ``out_dir``, preferring the local ml-journey copy.

    Args:
        out_dir: destination directory (created if missing).

    Returns:
        Path to the written CSV.
    """
    dest = out_dir / TITANIC_FILENAME
    if TITANIC_LOCAL_SOURCE.exists():
        logger.info("Copying local Titanic dataset from %s", TITANIC_LOCAL_SOURCE)
        shutil.copyfile(TITANIC_LOCAL_SOURCE, dest)
    else:
        logger.info("Downloading Titanic dataset from %s", TITANIC_URL)
        urllib.request.urlretrieve(TITANIC_URL, dest)  # noqa: S310 - fixed https URL
    _log_csv_stats(dest)
    return dest


def download_california(out_dir: Path) -> Path:
    """Fetch California Housing via sklearn and save it as one CSV with the target.

    Args:
        out_dir: destination directory (created if missing).

    Returns:
        Path to the written CSV (features + ``MedHouseVal`` target column).
    """
    from sklearn.datasets import fetch_california_housing

    logger.info("Fetching California Housing via sklearn (downloads on first use)")
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame  # features + MedHouseVal target
    dest = out_dir / CALIFORNIA_FILENAME
    frame.to_csv(dest, index=False)
    logger.info("Wrote %s: %d rows x %d columns", dest, frame.shape[0], frame.shape[1])
    return dest


def main() -> None:
    """Parse CLI arguments and download the requested dataset(s)."""
    parser = argparse.ArgumentParser(
        description="Download the demo datasets used by the template configs."
    )
    parser.add_argument(
        "--dataset",
        choices=["titanic", "california", "all"],
        default="all",
        help="Which dataset to fetch (default: all).",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output directory for the CSV files (default: data/raw).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("titanic", "all"):
        download_titanic(out_dir)
    if args.dataset in ("california", "all"):
        download_california(out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
