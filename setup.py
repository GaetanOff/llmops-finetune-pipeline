"""Setup script for LLMOps Fine-Tuning Pipeline."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="llmops-finetune-pipeline",
    version="1.0.0",
    description="Production-grade MLOps pipeline for fine-tuning and deploying LLMs",
    long_description=README,
    long_description_content_type="text/markdown",
    author="GaetanDev.fr",
    author_email="contact@gaetandev.fr",
    url="https://github.com/GaetanOff/llmops-finetune-pipeline",
    license="GPLv3",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "mlflow>=2.9.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "evidently>=0.4.10",
        "evaluate>=0.4.1",
        "rouge-score>=0.1.2",
        "nltk>=3.8.1",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "isort>=5.12.0",
        ],
        "dvc": [
            "dvc>=3.30.0",
            "dvc-s3>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="llm fine-tuning mlops lora transformers fastapi mlflow",
)
