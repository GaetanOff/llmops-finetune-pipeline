"""Tests for monitoring functionality."""

import pytest

from src.monitoring.quality_monitor import QualityMonitor


def test_quality_monitor_initialization():
    """Test quality monitor initialization."""
    monitor = QualityMonitor(window_size=50, log_to_mlflow=False)

    assert monitor.window_size == 50
    assert monitor.total_requests == 0
    assert monitor.failed_requests == 0


def test_log_prediction():
    """Test logging predictions."""
    monitor = QualityMonitor(window_size=10, log_to_mlflow=False)

    monitor.log_prediction(
        response_time=0.5,
        response_length=100,
        token_count=20,
        success=True,
    )

    assert monitor.total_requests == 1
    assert len(monitor.response_times) == 1
    assert monitor.response_times[0] == 0.5


def test_log_failed_prediction():
    """Test logging failed predictions."""
    monitor = QualityMonitor(window_size=10, log_to_mlflow=False)

    monitor.log_prediction(
        response_time=0.5,
        response_length=0,
        token_count=0,
        success=False,
    )

    assert monitor.total_requests == 1
    assert monitor.failed_requests == 1
    assert len(monitor.response_times) == 0


def test_get_metrics():
    """Test getting quality metrics."""
    monitor = QualityMonitor(window_size=10, log_to_mlflow=False)

    for i in range(5):
        monitor.log_prediction(
            response_time=0.5 + i * 0.1,
            response_length=100,
            token_count=20,
            success=True,
        )

    metrics = monitor.get_metrics()

    assert metrics["total_requests"] == 5
    assert metrics["failed_requests"] == 0
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_response_time"] > 0


def test_detect_anomalies():
    """Test anomaly detection."""
    monitor = QualityMonitor(window_size=100, log_to_mlflow=False)

    for i in range(20):
        monitor.log_prediction(
            response_time=0.5,
            response_length=100,
            token_count=20,
            success=True,
        )

    monitor.log_prediction(
        response_time=10.0,
        response_length=100,
        token_count=20,
        success=True,
    )

    anomalies = monitor.detect_anomalies(threshold_std=2.0)

    assert len(anomalies) > 0
    assert "Response time anomaly" in anomalies[0]
