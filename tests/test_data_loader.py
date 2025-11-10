"""Tests for data loading functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.dataset_loader import DatasetConfig, load_jsonl, save_jsonl


def test_load_jsonl():
    """Test loading JSONL files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        data = [
            {"text": "Sample text 1", "label": "positive"},
            {"text": "Sample text 2", "label": "negative"},
        ]
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        loaded_data = load_jsonl(temp_path)
        assert len(loaded_data) == 2
        assert loaded_data[0]["text"] == "Sample text 1"
        assert loaded_data[1]["label"] == "negative"
    finally:
        Path(temp_path).unlink()


def test_load_jsonl_with_limit():
    """Test loading JSONL files with max_samples limit."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(10):
            f.write(json.dumps({"text": f"Sample {i}"}) + "\n")
        temp_path = f.name

    try:
        loaded_data = load_jsonl(temp_path, max_samples=5)
        assert len(loaded_data) == 5
    finally:
        Path(temp_path).unlink()


def test_save_jsonl():
    """Test saving data to JSONL format."""
    data = [
        {"text": "Test 1", "value": 1},
        {"text": "Test 2", "value": 2},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.jsonl"
        save_jsonl(data, str(output_path))

        assert output_path.exists()

        loaded_data = load_jsonl(str(output_path))
        assert len(loaded_data) == 2
        assert loaded_data[0]["text"] == "Test 1"
        assert loaded_data[1]["value"] == 2


def test_dataset_config():
    """Test DatasetConfig dataclass."""
    config = DatasetConfig(
        train_file="train.jsonl",
        eval_file="eval.jsonl",
        test_file="test.jsonl",
        text_field="content",
        max_samples=100,
    )

    assert config.train_file == "train.jsonl"
    assert config.text_field == "content"
    assert config.max_samples == 100
    assert config.validation_split == 0.1
