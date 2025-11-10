"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_inference_service():
    """Create mock inference service."""
    mock_service = MagicMock()
    mock_service.model_loaded = True
    mock_service.model_path = "/models/test"
    mock_service.is_healthy.return_value = True
    mock_service.predict.return_value = {
        "generated_text": "Test response",
        "tokens_generated": 10,
        "inference_time": 0.5,
    }
    mock_service.get_metrics.return_value = {
        "total_predictions": 5,
        "average_response_time": 0.5,
        "average_tokens_generated": 10.0,
    }
    return mock_service


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@patch("src.api.app.inference_service")
def test_health_endpoint(mock_service, client):
    """Test health check endpoint."""
    mock_service.is_healthy.return_value = True
    mock_service.model_loaded = True
    mock_service.model_path = "/models/test"

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


@patch("src.api.app.inference_service")
def test_predict_endpoint(mock_service, client):
    """Test prediction endpoint."""
    mock_service.model_loaded = True
    mock_service.predict.return_value = {
        "generated_text": "Test response",
        "tokens_generated": 10,
        "inference_time": 0.5,
    }

    request_data = {
        "prompt": "Test prompt",
        "max_new_tokens": 100,
        "temperature": 0.7,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "tokens_generated" in data
    assert data["prompt"] == "Test prompt"


@patch("src.api.app.inference_service")
def test_metrics_endpoint(mock_service, client):
    """Test metrics endpoint."""
    mock_service.get_metrics.return_value = {
        "total_predictions": 5,
        "average_response_time": 0.5,
        "average_tokens_generated": 10.0,
    }

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "average_response_time" in data
