"""Data preprocessing utilities."""

import re
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocessor for cleaning and formatting training data."""

    def __init__(
            self,
            remove_urls: bool = True,
            remove_special_chars: bool = False,
            lowercase: bool = False,
            max_length: Optional[int] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            remove_urls: Whether to remove URLs
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert to lowercase
            max_length: Maximum text length (truncate if longer)
        """
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        if self.remove_special_chars:
            text = re.sub(r"[^\w\s.,!?-]", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def preprocess_example(self, example: Dict[str, str], text_field: str = "text") -> Dict[str, str]:
        """
        Preprocess a single example.

        Args:
            example: Dictionary containing text data
            text_field: Name of the text field

        Returns:
            Preprocessed example
        """
        if text_field in example:
            example[text_field] = self.clean_text(example[text_field])
        return example

    def preprocess_batch(
            self, examples: List[Dict[str, str]], text_field: str = "text"
    ) -> List[Dict[str, str]]:
        """
        Preprocess a batch of examples.

        Args:
            examples: List of examples
            text_field: Name of the text field

        Returns:
            List of preprocessed examples
        """
        return [self.preprocess_example(ex, text_field) for ex in examples]

    def format_instruction_data(
            self, instruction: str, input_text: str, output: str, template: Optional[str] = None
    ) -> str:
        """
        Format instruction-following data.

        Args:
            instruction: The instruction
            input_text: Optional input context
            output: Expected output
            template: Optional custom template

        Returns:
            Formatted text
        """
        if template is None:
            if input_text:
                template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            else:
                template = "### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return template.format(instruction=instruction, input=input_text, output=output)

    def create_qa_format(self, question: str, answer: str) -> str:
        """
        Format question-answer pairs.

        Args:
            question: The question
            answer: The answer

        Returns:
            Formatted text
        """
        return f"<|user|>\n{question}\n<|assistant|>\n{answer}"
