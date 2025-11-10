"""Quality monitoring for model outputs."""

import time
from collections import deque
from typing import Dict, List

import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityMonitor:
    """Monitor for tracking output quality metrics."""

    def __init__(self, window_size: int = 100, log_to_mlflow: bool = True):
        """
        Initialize the quality monitor.

        Args:
            window_size: Size of the sliding window for metrics
            log_to_mlflow: Whether to log metrics to MLflow
        """
        self.window_size = window_size
        self.log_to_mlflow = log_to_mlflow

        self.response_times: deque = deque(maxlen=window_size)
        self.response_lengths: deque = deque(maxlen=window_size)
        self.token_counts: deque = deque(maxlen=window_size)
        self.tokens_per_second: deque = deque(maxlen=window_size)

        self.total_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()

    def log_prediction(
            self,
            response_time: float,
            response_length: int,
            token_count: int,
            success: bool = True,
    ) -> None:
        """
        Log a prediction for quality monitoring.

        Args:
            response_time: Time taken for prediction
            response_length: Length of generated response
            token_count: Number of tokens generated
            success: Whether the prediction was successful
        """
        self.total_requests += 1

        if not success:
            self.failed_requests += 1
            return

        self.response_times.append(response_time)
        self.response_lengths.append(response_length)
        self.token_counts.append(token_count)

        if response_time > 0:
            self.tokens_per_second.append(token_count / response_time)

        if self.total_requests % self.window_size == 0:
            self._log_metrics()

    def _log_metrics(self) -> None:
        """Log aggregated metrics."""
        metrics = self.get_metrics()

        logger.info(
            f"Quality Metrics - Avg Response Time: {metrics['avg_response_time']:.2f}s, "
            f"Avg Tokens: {metrics['avg_tokens']:.1f}, "
            f"Throughput: {metrics['avg_tokens_per_second']:.2f} tok/s"
        )

        if self.log_to_mlflow:
            try:
                with mlflow.start_run():
                    mlflow.log_metrics(
                        {
                            "avg_response_time": metrics["avg_response_time"],
                            "avg_response_length": metrics["avg_response_length"],
                            "avg_tokens": metrics["avg_tokens"],
                            "avg_tokens_per_second": metrics["avg_tokens_per_second"],
                            "success_rate": metrics["success_rate"],
                            "total_requests": metrics["total_requests"],
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current quality metrics.

        Returns:
            Dictionary containing quality metrics
        """
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        avg_response_length = (
            sum(self.response_lengths) / len(self.response_lengths) if self.response_lengths else 0.0
        )
        avg_tokens = sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0.0
        avg_tokens_per_second = (
            sum(self.tokens_per_second) / len(self.tokens_per_second)
            if self.tokens_per_second
            else 0.0
        )

        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests
            if self.total_requests > 0
            else 0.0
        )

        uptime = time.time() - self.start_time

        return {
            "avg_response_time": avg_response_time,
            "avg_response_length": avg_response_length,
            "avg_tokens": avg_tokens,
            "avg_tokens_per_second": avg_tokens_per_second,
            "success_rate": success_rate,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "uptime_seconds": uptime,
        }

    def detect_anomalies(self, threshold_std: float = 3.0) -> List[str]:
        """
        Detect anomalies in quality metrics.

        Args:
            threshold_std: Number of standard deviations for anomaly detection

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if len(self.response_times) < 10:
            return anomalies

        import numpy as np

        response_times_array = np.array(self.response_times)
        mean_time = np.mean(response_times_array)
        std_time = np.std(response_times_array)

        recent_time = self.response_times[-1]
        if abs(recent_time - mean_time) > threshold_std * std_time:
            anomalies.append(
                f"Response time anomaly: {recent_time:.2f}s "
                f"(mean: {mean_time:.2f}s, std: {std_time:.2f}s)"
            )

        if self.total_requests > 0:
            failure_rate = self.failed_requests / self.total_requests
            if failure_rate > 0.1:
                anomalies.append(f"High failure rate: {failure_rate:.2%}")

        return anomalies
