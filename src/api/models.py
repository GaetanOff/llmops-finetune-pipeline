"""Pydantic models for API requests and responses."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    prompt: str = Field(..., description="Input prompt for the model")
    max_new_tokens: Optional[int] = Field(256, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Top-p sampling", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling", ge=1)
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    generated_text: str = Field(..., description="Generated text from the model")
    prompt: str = Field(..., description="Original input prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: Optional[str] = Field(None, description="Path to loaded model")
    gpu_available: bool = Field(..., description="Whether GPU is available")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    total_predictions: int = Field(..., description="Total number of predictions made")
    average_response_time: float = Field(..., description="Average response time in seconds")
    average_tokens_generated: float = Field(..., description="Average tokens generated per request")
