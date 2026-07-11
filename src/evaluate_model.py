"""Model evaluation module for the Enterprise Network Intrusion Intelligence System.

This module evaluates the persisted best model against the held-out test
set (``X_test_fe.csv`` / ``y_test.csv``). It reuses :class:`Predictor`
from ``src.inference.predictor`` to load artifacts and generate
predictions, ensuring evaluation exercises the exact same inference path
used in production. The test set is never used for feature selection,
training, or model comparison -- those steps are confined to
``feature_engineering.py`` and ``train_model.py``.

Computed metrics
-----------------
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix
- Full per-class classification report

Outputs (written under ``reports/``)
-------------------------------------
- metrics/evaluation_metrics.json : scalar metrics and classification report.
- metrics/confusion_matrix.csv : raw confusion matrix with class labels.
- figures/confusion_matrix.png : heatmap visualization of the confusion matrix.

Examples
--------
>>> from src.evaluate_model import ModelEvaluator
>>> evaluator = ModelEvaluator()
>>> results = evaluator.run()
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib

matplotlib.use("Agg")  # Headless backend; no display server available.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.inference.predictor import Predictor, PredictorError
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationError(Exception):
    """Base exception for all model-evaluation failures."""


class TestDataLoadError(EvaluationError):
    """Raised when the held-out test data cannot be loaded or is invalid."""


class ModelEvaluator:
    """Evaluates the persisted best model against the held-out test set.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing ``X_test_fe.csv`` and ``y_test.csv``. If
        ``None``, taken from ``Config.PROCESSED_DATA_DIR`` (falling back
        to ``"data/processed"``).
    model_dir : str or pathlib.Path, optional
        Directory holding the trained model artifacts, forwarded to
        :class:`Predictor`. If ``None``, ``Predictor`` resolves its own
        default.
    reports_dir : str or pathlib.Path, optional
        Root directory under which ``metrics/`` and ``figures/``
        subdirectories are created. If ``None``, taken from
        ``Config.REPORTS_DIR`` (falling back to ``"reports"``).

    Attributes
    ----------
    data_dir : pathlib.Path
        Resolved location of the held-out test data.
    metrics_dir : pathlib.Path
        Resolved location for JSON/CSV evaluation outputs.
    figures_dir : pathlib.Path
        Resolved location for evaluation plots.
    """

    _TARGET_COLUMN = "Attack Type"

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        model_dir: Union[str, Path, None] = None,
        reports_dir: Union[str, Path, None] = None,
    ) -> None:
        self.data_dir: Path = Path(
            data_dir if data_dir is not None else getattr(get_config(), "PROCESSED_DATA_DIR", "data/processed")
        )
        self._model_dir = model_dir
        reports_root = Path(
            reports_dir if reports_dir is not None else getattr(get_config(), "REPORTS_DIR", "reports")
        )
        self.metrics_dir: Path = reports_root / "metrics"
        self.figures_dir: Path = reports_root / "figures"

    def run(self) -> Dict[str, Any]:
        """Execute the full evaluation pipeline end to end.

        Returns
        -------
        dict
            Summary containing scalar metrics and the paths of the
            generated report artifacts.

        Raises
        ------
        TestDataLoadError
            If the test data cannot be loaded or is malformed.
        EvaluationError
            If model loading, prediction, or report generation fails.
        """
        logger.info("Starting model evaluation pipeline.")

        try:
            predictor = Predictor(model_dir=self._model_dir)
        except PredictorError as exc:
            message = f"Failed to load model artifacts for evaluation: {exc}"
            logger.error(message)
            raise EvaluationError(message) from exc

        X_test, y_true = self._load_test_data()

        try:
            y_pred = predictor.predict(X_test)
        except PredictorError as exc:
            message = f"Prediction on test set failed: {exc}"
            logger.error(message)
            raise EvaluationError(message) from exc

        class_labels = predictor.get_class_labels()
        scalar_metrics = self._compute_scalar_metrics(y_true, y_pred)
        report_dict = self._compute_classification_report(y_true, y_pred, class_labels)
        cm = self._compute_confusion_matrix(y_true, y_pred, class_labels)

        self._save_metrics_json(scalar_metrics, report_dict, predictor.model_info)
        self._save_confusion_matrix_csv(cm, class_labels)
        figure_path = self._save_confusion_matrix_figure(cm, class_labels)

        logger.info(
            "Evaluation complete. Accuracy=%.4f | Weighted F1=%.4f",
            scalar_metrics["accuracy"],
            scalar_metrics["f1_weighted"],
        )

        return {
            "metrics": scalar_metrics,
            "classification_report": report_dict,
            "confusion_matrix_path": str(self.metrics_dir / "confusion_matrix.csv"),
            "confusion_matrix_figure_path": str(figure_path),
            "metrics_json_path": str(self.metrics_dir / "evaluation_metrics.json"),
        }

    def _load_test_data(self) -> tuple:
        """Load ``X_test_fe.csv`` and ``y_test.csv`` exactly once each.

        Returns
        -------
        tuple of (pandas.DataFrame, pandas.Series)
            Test feature matrix and raw string target labels.

        Raises
        ------
        TestDataLoadError
            If either file is missing, empty, or row counts mismatch.
        """
        x_path = self.data_dir / "X_test_fe.csv"
        y_path = self.data_dir / "y_test.csv"

        if not x_path.is_file():
            raise TestDataLoadError(f"Required file not found: '{x_path}'.")
        if not y_path.is_file():
            raise TestDataLoadError(f"Required file not found: '{y_path}'.")

        try:
            X_test = pd.read_csv(x_path)
        except Exception as exc:
            raise TestDataLoadError(f"Failed to read '{x_path}': {exc}") from exc

        try:
            y_df = pd.read_csv(y_path)
        except Exception as exc:
            raise TestDataLoadError(f"Failed to read '{y_path}': {exc}") from exc

        if X_test.empty:
            raise TestDataLoadError(f"'{x_path}' contains no rows.")
        if y_df.empty:
            raise TestDataLoadError(f"'{y_path}' contains no rows.")
        if len(X_test) != len(y_df):
            raise TestDataLoadError(
                f"Row count mismatch between X_test_fe.csv ({len(X_test)}) and y_test.csv ({len(y_df)})."
            )
        if self._TARGET_COLUMN not in y_df.columns:
            raise TestDataLoadError(
                f"Target column '{self._TARGET_COLUMN}' not found in '{y_path}'. "
                f"Available columns: {list(y_df.columns)}"
            )

        numeric_columns = X_test.select_dtypes(include=[np.number]).columns
        X_test[numeric_columns] = X_test[numeric_columns].astype(np.float32)

        logger.info(
            "Loaded test data: X_test_fe.csv shape=%s, y_test.csv shape=%s.",
            X_test.shape,
            y_df.shape,
        )
        return X_test, y_df[self._TARGET_COLUMN]

    @staticmethod
    def _compute_scalar_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute accuracy and weighted precision/recall/F1.

        Parameters
        ----------
        y_true : pandas.Series
            Ground-truth decoded attack-type labels.
        y_pred : numpy.ndarray
            Predicted decoded attack-type labels.

        Returns
        -------
        dict
            Dictionary with keys ``accuracy``, ``precision_weighted``,
            ``recall_weighted``, and ``f1_weighted``.

        Raises
        ------
        EvaluationError
            If metric computation fails.
        """
        try:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_weighted": float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "recall_weighted": float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        except Exception as exc:
            raise EvaluationError(f"Failed to compute scalar metrics: {exc}") from exc

    @staticmethod
    def _compute_classification_report(
        y_true: pd.Series, y_pred: np.ndarray, class_labels: list
    ) -> Dict[str, Any]:
        """Compute the full per-class precision/recall/F1 report.

        Parameters
        ----------
        y_true : pandas.Series
            Ground-truth decoded attack-type labels.
        y_pred : numpy.ndarray
            Predicted decoded attack-type labels.
        class_labels : list of str
            Ordered class labels known to the model.

        Returns
        -------
        dict
            Output of ``sklearn.metrics.classification_report`` with
            ``output_dict=True``.

        Raises
        ------
        EvaluationError
            If report generation fails.
        """
        try:
            return classification_report(
                y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
            )
        except Exception as exc:
            raise EvaluationError(f"Failed to generate classification report: {exc}") from exc

    @staticmethod
    def _compute_confusion_matrix(
        y_true: pd.Series, y_pred: np.ndarray, class_labels: list
    ) -> np.ndarray:
        """Compute the confusion matrix ordered by ``class_labels``.

        Parameters
        ----------
        y_true : pandas.Series
            Ground-truth decoded attack-type labels.
        y_pred : numpy.ndarray
            Predicted decoded attack-type labels.
        class_labels : list of str
            Ordered class labels known to the model.

        Returns
        -------
        numpy.ndarray
            Confusion matrix of shape ``(n_classes, n_classes)``.

        Raises
        ------
        EvaluationError
            If computation fails.
        """
        try:
            return confusion_matrix(y_true, y_pred, labels=class_labels)
        except Exception as exc:
            raise EvaluationError(f"Failed to compute confusion matrix: {exc}") from exc

    def _save_metrics_json(
        self,
        scalar_metrics: Dict[str, float],
        report_dict: Dict[str, Any],
        model_info: Dict[str, Any],
    ) -> Path:
        """Persist scalar metrics and the classification report as JSON.

        Parameters
        ----------
        scalar_metrics : dict
            Accuracy and weighted precision/recall/F1.
        report_dict : dict
            Full per-class classification report.
        model_info : dict
            Training metadata of the evaluated model, included for
            traceability.

        Returns
        -------
        pathlib.Path
            Path of the written JSON file.

        Raises
        ------
        EvaluationError
            If the file cannot be written.
        """
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        destination = self.metrics_dir / "evaluation_metrics.json"
        payload = {
            "scalar_metrics": scalar_metrics,
            "classification_report": report_dict,
            "evaluated_model_info": model_info,
        }
        try:
            with open(destination, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
        except Exception as exc:
            raise EvaluationError(f"Failed to write '{destination}': {exc}") from exc

        logger.info("Saved evaluation metrics to '%s'.", destination)
        return destination

    def _save_confusion_matrix_csv(self, cm: np.ndarray, class_labels: list) -> Path:
        """Persist the confusion matrix as a labeled CSV.

        Parameters
        ----------
        cm : numpy.ndarray
            Confusion matrix of shape ``(n_classes, n_classes)``.
        class_labels : list of str
            Ordered class labels matching the matrix rows/columns.

        Returns
        -------
        pathlib.Path
            Path of the written CSV file.

        Raises
        ------
        EvaluationError
            If the file cannot be written.
        """
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        destination = self.metrics_dir / "confusion_matrix.csv"
        try:
            cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
            cm_df.to_csv(destination)
        except Exception as exc:
            raise EvaluationError(f"Failed to write '{destination}': {exc}") from exc

        logger.info("Saved confusion matrix CSV to '%s'.", destination)
        return destination

    def _save_confusion_matrix_figure(self, cm: np.ndarray, class_labels: list) -> Path:
        """Render and persist a heatmap visualization of the confusion matrix.

        Parameters
        ----------
        cm : numpy.ndarray
            Confusion matrix of shape ``(n_classes, n_classes)``.
        class_labels : list of str
            Ordered class labels matching the matrix rows/columns.

        Returns
        -------
        pathlib.Path
            Path of the written PNG file.

        Raises
        ------
        EvaluationError
            If the figure cannot be rendered or saved.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        destination = self.figures_dir / "confusion_matrix.png"

        try:
            n_classes = len(class_labels)
            fig_size = max(6, n_classes * 0.6)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(class_labels, rotation=90)
            ax.set_yticklabels(class_labels)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix - Best Model")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(destination, dpi=150)
            plt.close(fig)
        except Exception as exc:
            plt.close("all")
            raise EvaluationError(f"Failed to render confusion matrix figure: {exc}") from exc

        logger.info("Saved confusion matrix figure to '%s'.", destination)
        return destination


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()