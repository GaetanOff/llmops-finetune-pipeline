"""Metrics calculation for model evaluation."""

import math
from typing import Dict, List

import evaluate
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculator for various NLP metrics."""

    def __init__(self) -> None:
        """Initialize metrics calculators."""
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculate BLEU score.

        Args:
            predictions: List of predicted texts
            references: List of reference texts (each can have multiple references)

        Returns:
            Dictionary containing BLEU scores
        """
        results = self.bleu.compute(predictions=predictions, references=references)
        return {
            "bleu": results["bleu"],
            "bleu_1": results["precisions"][0] if len(results["precisions"]) > 0 else 0.0,
            "bleu_2": results["precisions"][1] if len(results["precisions"]) > 1 else 0.0,
            "bleu_3": results["precisions"][2] if len(results["precisions"]) > 2 else 0.0,
            "bleu_4": results["precisions"][3] if len(results["precisions"]) > 3 else 0.0,
        }

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dictionary containing ROUGE scores
        """
        results = self.rouge.compute(predictions=predictions, references=references)
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
            "rougeLsum": results["rougeLsum"],
        }

    def calculate_perplexity(
            self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, texts: List[str]
    ) -> float:
        """
        Calculate perplexity on a set of texts.

        Args:
            model: The language model
            tokenizer: The tokenizer
            texts: List of texts to evaluate

        Returns:
            Average perplexity
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

                if torch.cuda.is_available():
                    encodings = {k: v.cuda() for k, v in encodings.items()}

                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss
                num_tokens = encodings["input_ids"].size(1)

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return perplexity

    def calculate_all_metrics(
            self,
            predictions: List[str],
            references: List[str],
            model: PreTrainedModel = None,
            tokenizer: PreTrainedTokenizer = None,
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            model: Optional model for perplexity calculation
            tokenizer: Optional tokenizer for perplexity calculation

        Returns:
            Dictionary containing all metrics
        """
        metrics = {}

        logger.info("Calculating BLEU scores...")
        bleu_scores = self.calculate_bleu(predictions, [[ref] for ref in references])
        metrics.update(bleu_scores)

        logger.info("Calculating ROUGE scores...")
        rouge_scores = self.calculate_rouge(predictions, references)
        metrics.update(rouge_scores)

        if model is not None and tokenizer is not None:
            logger.info("Calculating perplexity...")
            perplexity = self.calculate_perplexity(model, tokenizer, predictions)
            metrics["perplexity"] = perplexity

        return metrics
