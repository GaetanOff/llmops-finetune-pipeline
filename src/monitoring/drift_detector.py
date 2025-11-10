"""Data drift detection using EvidentlyAI."""

from typing import Dict, List

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detector for monitoring data drift in model inputs and outputs."""

    def __init__(self, reference_window_size: int = 1000, detection_window_size: int = 100):
        """
        Initialize the drift detector.

        Args:
            reference_window_size: Size of reference data window
            detection_window_size: Size of detection data window
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.reference_data: List[Dict] = []
        self.current_data: List[Dict] = []

    def add_prediction(
            self,
            prompt: str,
            generated_text: str,
            tokens_generated: int,
            inference_time: float,
    ) -> None:
        """
        Add a prediction to the monitoring buffer.

        Args:
            prompt: Input prompt
            generated_text: Generated text
            tokens_generated: Number of tokens generated
            inference_time: Time taken for inference
        """
        record = {
            "prompt_length": len(prompt),
            "response_length": len(generated_text),
            "tokens_generated": tokens_generated,
            "inference_time": inference_time,
            "tokens_per_second": tokens_generated / inference_time if inference_time > 0 else 0,
        }

        if len(self.reference_data) < self.reference_window_size:
            self.reference_data.append(record)
        else:
            self.current_data.append(record)

            if len(self.current_data) >= self.detection_window_size:
                self.detect_drift()
                self.current_data = []

    def detect_drift(self) -> Dict[str, any]:
        """
        Detect drift between reference and current data.

        Returns:
            Dictionary containing drift detection results
        """
        if not self.reference_data or not self.current_data:
            logger.warning("Insufficient data for drift detection")
            return {"drift_detected": False, "message": "Insufficient data"}

        logger.info("Running drift detection...")

        reference_df = pd.DataFrame(self.reference_data)
        current_df = pd.DataFrame(self.current_data)

        column_mapping = ColumnMapping()

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        drift_results = report.as_dict()

        drift_detected = False
        if "metrics" in drift_results:
            for metric in drift_results["metrics"]:
                if "result" in metric and "drift_score" in metric["result"]:
                    drift_score = metric["result"]["drift_score"]
                    if drift_score > 0.5:
                        drift_detected = True
                        logger.warning(f"Drift detected with score: {drift_score:.4f}")

        results = {
            "drift_detected": drift_detected,
            "reference_size": len(self.reference_data),
            "current_size": len(self.current_data),
            "report": drift_results,
        }

        return results

    def save_report(self, output_path: str = "monitoring_report.html") -> None:
        """
        Save drift detection report to HTML.

        Args:
            output_path: Path to save the HTML report
        """
        if not self.reference_data or not self.current_data:
            logger.warning("Insufficient data to generate report")
            return

        reference_df = pd.DataFrame(self.reference_data)
        current_df = pd.DataFrame(self.current_data)

        column_mapping = ColumnMapping()

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])

        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        report.save_html(output_path)
        logger.info(f"Drift report saved to {output_path}")

    def reset_reference(self) -> None:
        """Reset the reference data with current data."""
        if len(self.current_data) >= self.reference_window_size:
            self.reference_data = self.current_data[-self.reference_window_size:]
            self.current_data = []
            logger.info("Reference data reset with current data")
