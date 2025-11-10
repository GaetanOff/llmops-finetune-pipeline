"""Dataset loading utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import PreTrainedTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    train_file: str
    eval_file: str
    test_file: Optional[str] = None
    text_field: str = "text"
    max_samples: Optional[int] = None
    validation_split: float = 0.1
    test_split: float = 0.1


def load_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file
        max_samples: Maximum number of samples to load

    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_dataset(
        config: DatasetConfig, tokenizer: Optional[PreTrainedTokenizer] = None
) -> Dict[str, Dataset]:
    """
    Load and prepare datasets for training.

    Args:
        config: Dataset configuration
        tokenizer: Optional tokenizer for preprocessing

    Returns:
        Dictionary containing train, eval, and optionally test datasets
    """
    logger.info(f"Loading datasets from {config.train_file}")

    datasets = {}

    train_path = Path(config.train_file)
    if train_path.suffix == ".jsonl":
        train_data = load_jsonl(config.train_file, config.max_samples)
        datasets["train"] = Dataset.from_list(train_data)
    else:
        datasets["train"] = hf_load_dataset(
            "json" if train_path.suffix == ".json" else "text",
            data_files=config.train_file,
            split="train",
        )

    eval_path = Path(config.eval_file)
    if eval_path.exists():
        if eval_path.suffix == ".jsonl":
            eval_data = load_jsonl(config.eval_file, config.max_samples)
            datasets["eval"] = Dataset.from_list(eval_data)
        else:
            datasets["eval"] = hf_load_dataset(
                "json" if eval_path.suffix == ".json" else "text",
                data_files=config.eval_file,
                split="train",
            )

    if config.test_file:
        test_path = Path(config.test_file)
        if test_path.exists():
            if test_path.suffix == ".jsonl":
                test_data = load_jsonl(config.test_file, config.max_samples)
                datasets["test"] = Dataset.from_list(test_data)
            else:
                datasets["test"] = hf_load_dataset(
                    "json" if test_path.suffix == ".json" else "text",
                    data_files=config.test_file,
                    split="train",
                )

    logger.info(f"Loaded {len(datasets['train'])} training samples")
    if "eval" in datasets:
        logger.info(f"Loaded {len(datasets['eval'])} evaluation samples")
    if "test" in datasets:
        logger.info(f"Loaded {len(datasets['test'])} test samples")

    return datasets


def tokenize_function(
        examples: Dict[str, List], tokenizer: PreTrainedTokenizer, text_field: str = "text"
) -> Dict[str, List]:
    """
    Tokenize examples for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        text_field: Name of the text field

    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples[text_field],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
    )
