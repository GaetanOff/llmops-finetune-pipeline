#!/usr/bin/env python
"""Evaluation script for assessing fine-tuned models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.utils.config_loader import load_env_config
from src.utils.logger import setup_logger

logger = setup_logger("evaluation", level="INFO", log_file="logs/evaluation.log")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLM")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (for PEFT models)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    load_env_config()

    logger.info(f"Loading model from {args.model_path}")
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )

    if args.interactive:
        logger.info("Starting interactive mode...")
        evaluator.interactive_test()
    else:
        logger.info(f"Evaluating on {args.test_file}")
        metrics = evaluator.evaluate_on_dataset(
            test_file=args.test_file,
            max_samples=args.max_samples,
            log_to_mlflow=not args.no_mlflow,
        )

        logger.info("Evaluation completed!")
        logger.info("Final metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
