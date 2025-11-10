"""FastAPI application for serving the fine-tuned model."""

import os
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceService
from src.api.models import (
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.config_loader import get_env_variable, load_env_config
from src.utils.logger import get_logger, setup_logger

load_env_config()
setup_logger(level=get_env_variable("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

inference_service: InferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global inference_service

    model_path = get_env_variable("MODEL_PATH", "./models/finetuned")
    base_model_name = get_env_variable("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    logger.info(f"Initializing inference service with model: {model_path}")
    inference_service = InferenceService(
        model_path=model_path, base_model_name=base_model_name, device="auto"
    )

    try:
        inference_service.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

    yield

    logger.info("Shutting down inference service")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="LLMOps Fine-tuned Model API",
        description="Production-ready API for serving fine-tuned LLMs",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "LLMOps Fine-tuned Model API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        if inference_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        return HealthResponse(
            status="healthy" if inference_service.is_healthy() else "unhealthy",
            model_loaded=inference_service.model_loaded,
            model_path=inference_service.model_path,
            gpu_available=torch.cuda.is_available(),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Generate predictions from the fine-tuned model.

        Args:
            request: Prediction request with prompt and generation parameters

        Returns:
            Generated text and metadata
        """
        if inference_service is None or not inference_service.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            result = inference_service.predict(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
            )

            return PredictionResponse(
                generated_text=result["generated_text"],
                prompt=request.prompt,
                tokens_generated=result["tokens_generated"],
                model_name=os.path.basename(inference_service.model_path),
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """
        Get inference metrics.

        Returns:
            Metrics about model performance
        """
        if inference_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        metrics = inference_service.get_metrics()
        return MetricsResponse(**metrics)

    return app


if __name__ == "__main__":
    import uvicorn

    host = get_env_variable("API_HOST", "0.0.0.0")
    port = int(get_env_variable("API_PORT", "8000"))
    reload = get_env_variable("API_RELOAD", "false").lower() == "true"

    uvicorn.run("src.api.app:create_app", host=host, port=port, reload=reload, factory=True)
