#!/usr/bin/env python
"""Script to generate sample dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_generator import generate_sample_dataset
from src.utils.logger import setup_logger

logger = setup_logger("data_generation", level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate sample Q&A dataset")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )

    return parser.parse_args()


def main() -> None:
    """Main data generation function."""
    args = parse_args()

    logger.info("Generating sample dataset...")
    generate_sample_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        train_split=args.train_split,
    )
    logger.info("Dataset generation completed!")


if __name__ == "__main__":
    main()
