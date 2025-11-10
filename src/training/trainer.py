"""Main training module for fine-tuning LLMs with LoRA."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from src.data.dataset_loader import DatasetConfig, load_dataset, tokenize_function
from src.training.lora_config import create_lora_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./models/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"


class LLMTrainer:
    """Trainer class for fine-tuning LLMs with LoRA and MLflow tracking."""

    def __init__(self, config: TrainingConfig, mlflow_experiment_name: str = "llmops-finetuning"):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            mlflow_experiment_name: MLflow experiment name
        """
        self.config = config
        self.mlflow_experiment_name = mlflow_experiment_name
        self.model = None
        self.tokenizer = None
        self.trainer = None

        mlflow.set_experiment(mlflow_experiment_name)

    def load_model_and_tokenizer(self) -> None:
        """Load and prepare the model and tokenizer for training."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.model_max_length = self.config.max_length

        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, device_map="auto", trust_remote_code=True
            )

        lora_config = create_lora_config(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("Model and tokenizer loaded successfully")

    def prepare_datasets(self, dataset_config: DatasetConfig) -> Dict:
        """
        Load and prepare datasets for training.

        Args:
            dataset_config: Dataset configuration

        Returns:
            Dictionary containing tokenized datasets
        """
        logger.info("Loading and preparing datasets...")

        datasets = load_dataset(dataset_config, self.tokenizer)

        tokenized_datasets = {}
        for split, dataset in datasets.items():
            tokenized_datasets[split] = dataset.map(
                lambda examples: tokenize_function(examples, self.tokenizer, dataset_config.text_field),
                batched=True,
                remove_columns=dataset.column_names,
            )

        logger.info("Datasets prepared successfully")
        return tokenized_datasets

    def train(self, dataset_config: DatasetConfig) -> None:
        """
        Execute the training process with MLflow tracking.

        Args:
            dataset_config: Dataset configuration
        """
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model_name": self.config.model_name,
                    "num_epochs": self.config.num_train_epochs,
                    "batch_size": self.config.per_device_train_batch_size,
                    "learning_rate": self.config.learning_rate,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "max_length": self.config.max_length,
                }
            )

            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()

            tokenized_datasets = self.prepare_datasets(dataset_config)

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type=self.config.lr_scheduler_type,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                fp16=self.config.fp16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim=self.config.optim,
                max_grad_norm=self.config.max_grad_norm,
                report_to=["mlflow"],
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets.get("eval"),
                tokenizer=self.tokenizer,
            )

            logger.info("Starting training...")
            train_result = self.trainer.train()

            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            logger.info("Training completed successfully")

            self.save_model()

    def save_model(self, output_path: Optional[str] = None) -> None:
        """
        Save the fine-tuned model.

        Args:
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "final_model")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_path}")
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)

        mlflow.log_artifacts(output_path, artifact_path="model")
        logger.info("Model saved successfully")
