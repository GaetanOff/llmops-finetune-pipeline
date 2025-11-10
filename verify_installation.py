#!/usr/bin/env python
"""Verify that the LLMOps pipeline is correctly installed."""

import sys
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING: {file_path}")
        return False


def check_directory_exists(dir_path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(dir_path).is_dir():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING: {dir_path}")
        return False


def check_imports() -> bool:
    """Check if key modules can be imported."""
    print("\nChecking Python imports...")
    all_good = True
    
    modules = [
        ("src.data.dataset_loader", "Data loader module"),
        ("src.training.trainer", "Training module"),
        ("src.evaluation.evaluator", "Evaluation module"),
        ("src.api.app", "API module"),
        ("src.monitoring.drift_detector", "Monitoring module"),
    ]
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"✗ {description} - ERROR: {e}")
            all_good = False
    
    return all_good


def main() -> int:
    """Main verification function."""
    print("=" * 60)
    print("LLMOps Fine-Tuning Pipeline - Installation Verification")
    print("=" * 60)
    
    all_checks_passed = True
    
    print("\nChecking core source files...")
    files_to_check = [
        ("src/__init__.py", "Main package init"),
        ("src/data/dataset_loader.py", "Dataset loader"),
        ("src/data/data_preprocessor.py", "Data preprocessor"),
        ("src/training/trainer.py", "Training module"),
        ("src/evaluation/evaluator.py", "Evaluation module"),
        ("src/api/app.py", "API application"),
        ("src/monitoring/drift_detector.py", "Drift detector"),
    ]
    
    for file_path, description in files_to_check:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking configuration files...")
    config_files = [
        ("configs/model_config.yaml", "Model configuration"),
        ("configs/api_config.yaml", "API configuration"),
        (".env.example", "Environment template"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    for file_path, description in config_files:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking scripts...")
    scripts = [
        ("scripts/train.py", "Training script"),
        ("scripts/evaluate.py", "Evaluation script"),
        ("scripts/generate_data.py", "Data generation script"),
    ]
    
    for file_path, description in scripts:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking tests...")
    test_files = [
        ("tests/test_data_loader.py", "Data loader tests"),
        ("tests/test_api.py", "API tests"),
    ]
    
    for file_path, description in test_files:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking Docker files...")
    docker_files = [
        ("Dockerfile", "Docker image definition"),
        ("docker-compose.yaml", "Docker compose config"),
    ]
    
    for file_path, description in docker_files:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking documentation...")
    docs = [
        ("README.md", "Main documentation"),
        ("ARCHITECTURE.md", "Architecture docs"),
        ("CONTRIBUTING.md", "Contributing guide"),
        ("QUICKSTART.md", "Quick start guide"),
    ]
    
    for file_path, description in docs:
        all_checks_passed &= check_file_exists(file_path, description)
    
    print("\nChecking directory structure...")
    directories = [
        ("src/data", "Data module"),
        ("src/training", "Training module"),
        ("src/evaluation", "Evaluation module"),
        ("src/api", "API module"),
        ("src/monitoring", "Monitoring module"),
        ("tests", "Test suite"),
        ("configs", "Configurations"),
        ("scripts", "Executable scripts"),
    ]
    
    for dir_path, description in directories:
        all_checks_passed &= check_directory_exists(dir_path, description)
    
    # Check imports
    all_checks_passed &= check_imports()
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All checks passed! Installation verified.")
        print("\nNext steps:")
        print("  1. Generate sample data: make generate-data")
        print("  2. Train a model: make train")
        print("  3. Start API: make api")
        print("  4. See QUICKSTART.md for detailed guide")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
