#!/usr/bin/env python
"""Training script for fine-tuning LLMs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetConfig
from src.training.trainer import LLMTrainer, TrainingConfig
from src.utils.config_loader import load_config, load_env_config
from src.utils.logger import setup_logger

logger = setup_logger("training", level="INFO", log_file="logs/training.log")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a fine-tuned LLM with LoRA")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()
    load_env_config()

    logger.info("Loading configuration...")
    config_dict = load_config(args.config)

    model_config = config_dict.get("model", {})
    training_config_dict = config_dict.get("training", {})
    lora_config = config_dict.get("lora", {})
    dataset_config_dict = config_dict.get("dataset", {})

    training_config = TrainingConfig(
        model_name=args.model_name or model_config.get("name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        output_dir=training_config_dict.get("output_dir", "./models/checkpoints"),
        num_train_epochs=args.epochs or training_config_dict.get("num_train_epochs", 3),
        per_device_train_batch_size=args.batch_size or training_config_dict.get("per_device_train_batch_size", 4),
        learning_rate=args.learning_rate or training_config_dict.get("learning_rate", 2e-4),
        lora_r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        max_length=model_config.get("max_length", 512),
    )

    dataset_config = DatasetConfig(
        train_file=dataset_config_dict.get("train_file", "data/processed/train.jsonl"),
        eval_file=dataset_config_dict.get("eval_file", "data/processed/eval.jsonl"),
        test_file=dataset_config_dict.get("test_file", "data/processed/test.jsonl"),
        text_field=dataset_config_dict.get("text_field", "text"),
        max_samples=dataset_config_dict.get("max_samples"),
    )

    logger.info("Initializing trainer...")
    trainer = LLMTrainer(training_config)

    logger.info("Starting training...")
    try:
        trainer.train(dataset_config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
