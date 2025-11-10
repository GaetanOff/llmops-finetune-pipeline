"""LoRA configuration utilities."""

from typing import List, Optional

from peft import LoraConfig, TaskType

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_lora_config(
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.

    Args:
        r: LoRA rank (dimensionality of the low-rank matrices)
        lora_alpha: LoRA scaling factor
        target_modules: List of module names to apply LoRA to
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias training strategy ('none', 'all', or 'lora_only')
        task_type: Type of task (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)

    Returns:
        Configured LoraConfig object
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    logger.info(f"Creating LoRA config with r={r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {target_modules}")

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM,
    )

    return config
