"""Tests for data preprocessing."""

import pytest

from src.data.data_preprocessor import DataPreprocessor


def test_clean_text_basic():
    """Test basic text cleaning."""
    preprocessor = DataPreprocessor()

    text = "  This is   a test.  "
    cleaned = preprocessor.clean_text(text)

    assert cleaned == "This is a test."


def test_remove_urls():
    """Test URL removal."""
    preprocessor = DataPreprocessor(remove_urls=True)

    text = "Check out https://example.com for more info."
    cleaned = preprocessor.clean_text(text)

    assert "https://example.com" not in cleaned
    assert "Check out" in cleaned


def test_lowercase():
    """Test lowercase conversion."""
    preprocessor = DataPreprocessor(lowercase=True)

    text = "This Is A Test"
    cleaned = preprocessor.clean_text(text)

    assert cleaned == "this is a test"


def test_max_length():
    """Test text truncation."""
    preprocessor = DataPreprocessor(max_length=10)

    text = "This is a very long text that should be truncated"
    cleaned = preprocessor.clean_text(text)

    assert len(cleaned) == 10


def test_preprocess_example():
    """Test preprocessing a single example."""
    preprocessor = DataPreprocessor()

    example = {"text": "  Hello World  ", "label": "test"}
    processed = preprocessor.preprocess_example(example)

    assert processed["text"] == "Hello World"
    assert processed["label"] == "test"


def test_format_instruction_data():
    """Test instruction data formatting."""
    preprocessor = DataPreprocessor()

    instruction = "Translate to French"
    input_text = "Hello"
    output = "Bonjour"

    formatted = preprocessor.format_instruction_data(instruction, input_text, output)

    assert "Instruction:" in formatted
    assert "Input:" in formatted
    assert "Response:" in formatted
    assert instruction in formatted
    assert output in formatted


def test_create_qa_format():
    """Test Q&A formatting."""
    preprocessor = DataPreprocessor()

    question = "What is AI?"
    answer = "Artificial Intelligence"

    formatted = preprocessor.create_qa_format(question, answer)

    assert "<|user|>" in formatted
    assert "<|assistant|>" in formatted
    assert question in formatted
    assert answer in formatted
