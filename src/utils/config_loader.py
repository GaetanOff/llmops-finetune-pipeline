"""Configuration loader utility."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_env_config() -> None:
    """Load environment variables from .env file."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def get_env_variable(var_name: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional default and validation.

    Args:
        var_name: Name of the environment variable
        default: Default value if variable not found
        required: Whether the variable is required

    Returns:
        Value of the environment variable

    Raises:
        ValueError: If required variable is not set
    """
    value = os.getenv(var_name, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{var_name}' is not set")

    return value
