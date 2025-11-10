#!/usr/bin/env python
"""Script to run monitoring and generate drift reports."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift_detector import DriftDetector
from src.utils.logger import setup_logger

logger = setup_logger("monitoring", level="INFO")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run monitoring and drift detection")

    parser.add_argument(
        "--output",
        type=str,
        default="monitoring_report.html",
        help="Output path for monitoring report",
    )

    return parser.parse_args()


def main() -> None:
    """Main monitoring function."""
    args = parse_args()

    logger.info("Initializing drift detector...")
    detector = DriftDetector()

    logger.info("Generating monitoring report...")
    detector.save_report(args.output)
    logger.info(f"Monitoring report saved to {args.output}")


if __name__ == "__main__":
    main()
