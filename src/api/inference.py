"""Inference service for model predictions."""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Service for running model inference."""

    def __init__(
            self,
            model_path: str,
            base_model_name: Optional[str] = None,
            device: str = "auto",
    ):
        """
        Initialize the inference service.

        Args:
            model_path: Path to the fine-tuned model
            base_model_name: Optional base model name for PEFT models
            device: Device to run inference on
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.total_tokens_generated = 0

    def load_model(self) -> None:
        """Load the model and tokenizer into memory."""
        if self.model_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dir = Path(self.model_path)
        if (model_dir / "adapter_config.json").exists():
            if self.base_model_name is None:
                raise ValueError("base_model_name required for PEFT models")

            logger.info(f"Loading PEFT model with base model: {self.base_model_name}")
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
        self.model_loaded = True
        logger.info("Model loaded successfully")

    def predict(
            self,
            prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
    ) -> Dict[str, any]:
        """
        Generate prediction for a given prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Dictionary containing generated text and metadata
        """
        if not self.model_loaded:
            self.load_model()

        start_time = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()

        inference_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])

        self.prediction_count += 1
        self.total_inference_time += inference_time
        self.total_tokens_generated += tokens_generated

        logger.info(
            f"Generated {tokens_generated} tokens in {inference_time:.2f}s "
            f"({tokens_generated / inference_time:.2f} tokens/s)"
        )

        return {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "inference_time": inference_time,
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Get inference metrics.

        Returns:
            Dictionary containing metrics
        """
        avg_time = (
            self.total_inference_time / self.prediction_count if self.prediction_count > 0 else 0.0
        )
        avg_tokens = (
            self.total_tokens_generated / self.prediction_count if self.prediction_count > 0 else 0.0
        )

        return {
            "total_predictions": self.prediction_count,
            "average_response_time": avg_time,
            "average_tokens_generated": avg_tokens,
        }

    def is_healthy(self) -> bool:
        """
        Check if the service is healthy.

        Returns:
            True if service is healthy
        """
        return self.model_loaded
