"""Model evaluator for comprehensive assessment."""

from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset_loader import load_jsonl
from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluator for assessing fine-tuned models."""

    def __init__(
            self,
            model_path: str,
            base_model_name: Optional[str] = None,
            device: str = "auto",
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the fine-tuned model
            base_model_name: Optional base model name for PEFT models
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.metrics_calculator = MetricsCalculator()

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dir = Path(self.model_path)
        if (model_dir / "adapter_config.json").exists():
            if self.base_model_name is None:
                raise ValueError("base_model_name required for PEFT models")

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate_predictions(
            self,
            prompts: List[str],
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate predictions for a list of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            List of generated texts
        """
        if self.model is None:
            self.load_model()

        predictions = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            predictions.append(generated_text)

        return predictions

    def evaluate_on_dataset(
            self,
            test_file: str,
            text_field: str = "text",
            max_samples: Optional[int] = None,
            log_to_mlflow: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.

        Args:
            test_file: Path to test dataset
            text_field: Name of the text field
            max_samples: Maximum number of samples to evaluate
            log_to_mlflow: Whether to log results to MLflow

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on {test_file}")

        test_data = load_jsonl(test_file, max_samples)

        prompts = []
        references = []

        for item in test_data:
            if "question" in item and "answer" in item:
                prompts.append(f"<|user|>\n{item['question']}\n<|assistant|>\n")
                references.append(item["answer"])
            elif text_field in item:
                parts = item[text_field].split("<|assistant|>")
                if len(parts) == 2:
                    prompts.append(parts[0] + "<|assistant|>\n")
                    references.append(parts[1].strip())

        logger.info(f"Generating predictions for {len(prompts)} examples...")
        predictions = self.generate_predictions(prompts)

        logger.info("Calculating metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, references, self.model, self.tokenizer
        )

        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        if log_to_mlflow:
            with mlflow.start_run():
                mlflow.log_metrics(metrics)
                mlflow.log_param("test_file", test_file)
                mlflow.log_param("num_samples", len(prompts))

        return metrics

    def interactive_test(self) -> None:
        """Run interactive testing session."""
        if self.model is None:
            self.load_model()

        logger.info("Interactive testing mode. Type 'quit' to exit.")

        while True:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break

            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            predictions = self.generate_predictions([formatted_prompt], max_new_tokens=256)

            print(f"\nGenerated response:\n{predictions[0]}\n")
