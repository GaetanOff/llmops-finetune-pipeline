.PHONY: install test lint format clean docker-build docker-up docker-down train evaluate api help

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make train        - Run training"
	@echo "  make evaluate     - Run evaluation"
	@echo "  make api          - Start API server"
	@echo "  make generate-data - Generate sample dataset"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src tests
	black --check src tests
	mypy src --ignore-missing-imports

format:
	black src tests
	isort src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist

docker-build:
	docker build -t llmops-api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

train:
	python scripts/train.py --config configs/model_config.yaml

evaluate:
	python scripts/evaluate.py \
		--model-path models/checkpoints/final_model \
		--base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
		--test-file data/processed/test.jsonl

api:
	uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000 --factory --reload

generate-data:
	python scripts/generate_data.py --num-samples 100

mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 0.0.0.0 \
		--port 5000

monitoring:
	python scripts/run_monitoring.py --output monitoring_report.html
