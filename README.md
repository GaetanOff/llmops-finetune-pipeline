# LLMOps Fine-Tuning Pipeline

A production-grade, end-to-end MLOps pipeline for fine-tuning, evaluating, and deploying small open-source Large Language Models using LoRA (Low-Rank Adaptation).

## Overview

This repository implements a complete LLMOps workflow with:

- **Parameter-Efficient Fine-Tuning**: LoRA-based fine-tuning with 4-bit quantization
- **Experiment Tracking**: MLflow integration for comprehensive experiment management
- **Model Versioning**: DVC for data and model version control
- **Production API**: FastAPI-based REST API for model serving
- **Monitoring & Observability**: Data drift detection and quality monitoring
- **CI/CD Pipeline**: Automated testing, linting, and deployment
- **Containerization**: Docker and docker-compose for reproducible environments

## Architecture

```
┌─────────────────┐
│  Data Pipeline  │
│  - Generation   │
│  - Preprocessing│
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Training Module │─────▶│   MLflow     │
│  - LoRA Config  │      │  Tracking    │
│  - Fine-tuning  │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Evaluation    │─────▶│     DVC      │
│  - Metrics      │      │   Versioning │
│  - BLEU/ROUGE   │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  FastAPI Server │◀────▶│  Monitoring  │
│  - /predict     │      │  - Drift     │
│  - /health      │      │  - Quality   │
└─────────────────┘      └──────────────┘
```

## Features

### 1. Model Training
- Fine-tune TinyLlama, Mistral-7B, or Phi-3 models
- LoRA for parameter-efficient training
- 4-bit quantization with bitsandbytes
- Gradient checkpointing for memory efficiency
- MLflow experiment tracking

### 2. Evaluation Pipeline
- Automatic metric computation (BLEU, ROUGE, Perplexity)
- Interactive testing mode
- Model comparison across experiments

### 3. Model Serving
- REST API with FastAPI
- Health checks and metrics endpoints
- Configurable inference parameters
- GPU acceleration support

### 4. Monitoring
- Data drift detection with EvidentlyAI
- Quality metrics tracking
- Anomaly detection
- MLflow integration

### 5. MLOps Automation
- CI/CD with GitHub Actions
- Automated testing and linting
- Docker containerization
- Model retraining on data changes

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Docker and docker-compose (for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/GaetanOff/llmops-finetune-pipeline
cd llmops-finetune-pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize DVC:
```bash
dvc init
```

## Quick Start

### 1. Generate Sample Dataset
```bash
python scripts/generate_data.py --num-samples 100
```

### 2. Train Model
```bash
python scripts/train.py --config configs/model_config.yaml
```

### 3. Evaluate Model
```bash
python scripts/evaluate.py \
    --model-path models/checkpoints/final_model \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --test-file data/processed/test.jsonl
```

### 4. Start API Server
```bash
uvicorn src.api.app:create_app --host 0.0.0.0 --port 8000 --factory
```

### 5. Test API
```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is machine learning?",
        "max_new_tokens": 256,
        "temperature": 0.7
    }'
```

## Docker Deployment

### Using Docker Compose

1. Build and start services:
```bash
docker-compose up -d
```

This starts:
- MLflow server on port 5000
- API server on port 8000

2. Access services:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

3. Stop services:
```bash
docker-compose down
```

### Building Docker Image

```bash
docker build -t llmops-api:latest .
```

## Project Structure

```
llmops-finetune-pipeline/
├── src/
│   ├── data/               # Data processing modules
│   │   ├── dataset_loader.py
│   │   ├── data_preprocessor.py
│   │   └── data_generator.py
│   ├── training/           # Training modules
│   │   ├── trainer.py
│   │   └── lora_config.py
│   ├── evaluation/         # Evaluation modules
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── api/               # API serving
│   │   ├── app.py
│   │   ├── inference.py
│   │   └── models.py
│   ├── monitoring/        # Monitoring modules
│   │   ├── drift_detector.py
│   │   └── quality_monitor.py
│   └── utils/             # Utility functions
│       ├── config_loader.py
│       └── logger.py
├── configs/               # Configuration files
│   ├── model_config.yaml
│   ├── api_config.yaml
│   └── monitoring_config.yaml
├── tests/                 # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_api.py
│   └── test_monitoring.py
├── scripts/              # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_data.py
│   └── run_monitoring.py
├── .github/workflows/    # CI/CD workflows
│   ├── ci.yml
│   ├── docker-build.yml
│   └── retrain.yml
├── data/                 # Data directory
│   ├── processed/
│   └── raw/
├── models/               # Model storage
│   ├── checkpoints/
│   └── finetuned/
├── docker-compose.yaml   # Docker orchestration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  max_length: 512

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  batch_size: 4
```

### Environment Variables (`.env`)

```bash
# Model Configuration
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODEL_PATH=./models/finetuned

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## MLflow Experiment Tracking

### Start MLflow Server
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000
```

### View Experiments
Navigate to http://localhost:5000 to view:
- Training metrics and losses
- Model parameters and hyperparameters
- Model artifacts and checkpoints
- Evaluation metrics

## DVC Model Versioning

### Track Model with DVC
```bash
dvc add models/checkpoints/final_model
git add models/checkpoints/final_model.dvc .gitignore
git commit -m "Add trained model"
```

### Push to Remote Storage
```bash
dvc push
```

### Pull Model from Storage
```bash
dvc pull
```

## API Documentation

### Endpoints

#### `POST /predict`
Generate predictions from the fine-tuned model.

**Request:**
```json
{
    "prompt": "What is deep learning?",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
}
```

**Response:**
```json
{
    "generated_text": "Deep learning is a subset of machine learning...",
    "prompt": "What is deep learning?",
    "tokens_generated": 45,
    "model_name": "final_model"
}
```

#### `GET /health`
Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_path": "./models/finetuned",
    "gpu_available": true
}
```

#### `GET /metrics`
Get inference metrics.

**Response:**
```json
{
    "total_predictions": 150,
    "average_response_time": 0.45,
    "average_tokens_generated": 42.3
}
```

## Monitoring

### Data Drift Detection

The system automatically monitors for data drift using EvidentlyAI:

```python
from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector()
detector.add_prediction(prompt, response, tokens, time)
drift_results = detector.detect_drift()
```

### Quality Monitoring

Track model output quality:

```python
from src.monitoring.quality_monitor import QualityMonitor

monitor = QualityMonitor()
monitor.log_prediction(response_time, length, tokens, success=True)
metrics = monitor.get_metrics()
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test Suite
```bash
pytest tests/test_api.py -v
```

## CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Linting (black, flake8, mypy)
   - Unit tests with coverage
   - Security scanning

2. **Docker Build** (`.github/workflows/docker-build.yml`)
   - Build Docker image
   - Push to GitHub Container Registry
   - Tag with version/commit SHA

3. **Auto-Retrain** (`.github/workflows/retrain.yml`)
   - Triggers on data changes
   - Runs training pipeline
   - Pushes model with DVC

### Triggering Workflows

```bash
# Trigger retrain on data change
git add data/processed/train.jsonl
git commit -m "Update training data [retrain]"
git push
```

## Performance Optimization

### Memory Optimization
- 4-bit quantization reduces memory by ~75%
- Gradient checkpointing for large models
- LoRA reduces trainable parameters by ~90%

### Training Speed
- Mixed precision training (FP16)
- Gradient accumulation for effective larger batches
- Optimized data loading with prefetch

### Inference Optimization
- Model caching and warm-up
- Batch inference support
- GPU acceleration

## Best Practices

1. **Data Quality**: Always validate and clean your training data
2. **Experiment Tracking**: Log all hyperparameters and metrics to MLflow
3. **Model Versioning**: Use DVC for all model checkpoints
4. **Testing**: Write tests for data processing and API endpoints
5. **Monitoring**: Set up alerts for drift detection and quality degradation
6. **Documentation**: Keep configs and code well-documented

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Enable gradient checkpointing
- Use smaller LoRA rank (r)
- Increase gradient accumulation steps

### Slow Training
- Enable mixed precision (FP16)
- Use faster optimizer (adamw_8bit)
- Reduce logging frequency
- Use smaller validation set

### API Errors
- Check model path is correct
- Ensure GPU is available
- Verify environment variables
- Check logs in `logs/` directory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

All the code is licensed under GPL v3.
Feel free to modify and improve it!

## Acknowledgments

- Hugging Face Transformers and PEFT libraries
- MLflow for experiment tracking
- EvidentlyAI for monitoring
- FastAPI for API framework
- The open-source LLM community

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{llmops_pipeline,
  title = {LLMOps Fine-Tuning Pipeline},
  author = {GaetanDev.fr},
  year = {2025},
  url = {https://github.com/GaetanOff/llmops-finetune-pipeline}
}
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: contact@gaetandev.fr

## Roadmap

- [ ] Add support for more base models (Llama-3, Gemma)
- [ ] Implement RLHF pipeline
- [ ] Add multi-GPU training support
- [ ] Integration with cloud providers (AWS, GCP, Azure)
- [ ] Web UI for monitoring and management
- [ ] Advanced prompt engineering tools
- [ ] Model quantization (GPTQ, AWQ)
- [ ] Kubernetes deployment manifests
