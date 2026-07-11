"""Model training pipeline for the Enterprise Network Intrusion Intelligence System.

This module trains exactly two candidate models -- a Random Forest and an
XGBoost classifier, each with fixed, non-tuned hyperparameters -- on the
engineered feature set (``X_train_fe.csv`` / ``y_train.csv``). To select a
winner without ever touching the held-out test set, a stratified
validation split is carved out of the training data. Each candidate is
trained exactly once on the remaining training rows and scored on that
validation split using the weighted F1-score. The higher-scoring model
(weighted precision as tiebreaker) is persisted; the losing model is
discarded and never written to disk.

All artifacts are written atomically: they are first serialized to
temporary files and only moved into their final location after every
prior step has succeeded, guaranteeing that ``models/best_model.pkl`` is
never left empty or corrupted if training or evaluation raises.

Outputs (written to ``models/``)
---------------------------------
- best_model.pkl : the winning fitted estimator.
- label_encoder.pkl : fitted ``LabelEncoder`` for the "Attack Type" target.
- selected_features.csv : ordered feature names used by the model.
- training_metadata.json : winning model name, hyperparameters, and
  validation metrics.

Examples
--------
>>> from src.train_model import ModelTrainer
>>> trainer = ModelTrainer()
>>> summary = trainer.run()
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingError(Exception):
    """Base exception for all model-training pipeline failures."""


class DataLoadError(TrainingError):
    """Raised when the engineered training data cannot be loaded or is invalid."""


class ArtifactSaveError(TrainingError):
    """Raised when a trained artifact fails to persist to disk."""


# Fixed, non-tuned hyperparameters. No parameter grid or search is used.
RANDOM_FOREST_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

VALIDATION_FRACTION: float = 0.1
RANDOM_STATE: int = 42


class ModelTrainer:
    """Trains, compares, and persists the intrusion-detection model.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing ``X_train_fe.csv`` and ``y_train.csv``. If
        ``None``, taken from ``Config.PROCESSED_DATA_DIR`` (falling back
        to ``"data/processed"``).
    model_dir : str or pathlib.Path, optional
        Directory to which the winning model artifacts are written. If
        ``None``, taken from ``Config.MODELS_DIR`` (falling back to
        ``"models"``).
    validation_fraction : float, default=0.1
        Fraction of the training data held out, via stratified split, to
        score and compare the two candidate models.
    random_state : int, default=42
        Seed used for the train/validation split.

    Attributes
    ----------
    data_dir : pathlib.Path
        Resolved location of the engineered training data.
    model_dir : pathlib.Path
        Resolved location where artifacts are persisted.
    """

    _TARGET_COLUMN = "Attack Type"

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        model_dir: Union[str, Path, None] = None,
        validation_fraction: float = VALIDATION_FRACTION,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.data_dir: Path = Path(
            data_dir if data_dir is not None else getattr(get_config(), "PROCESSED_DATA_DIR", "data/processed")
        )
        self.model_dir: Path = Path(
            model_dir if model_dir is not None else getattr(get_config(), "MODELS_DIR", "models")
        )
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def run(self) -> Dict[str, Any]:
        """Execute the full training pipeline end to end.

        Returns
        -------
        dict
            Summary containing the winning model's name and its
            validation metrics.

        Raises
        ------
        DataLoadError
            If the engineered training data cannot be loaded or is
            malformed.
        TrainingError
            If model fitting fails.
            ArtifactSaveError is raised, and re-raised as such, if
            persistence fails.
        """
        start_time = time.time()
        logger.info("Starting model training pipeline.")

        X, y_raw, feature_names = self._load_data()
        label_encoder, y_encoded = self._encode_labels(y_raw)

        X_train_split, X_val_split, y_train_split, y_val_split = self._split(X, y_encoded)
        # Free the full-size intermediate frame as soon as it is no longer needed.
        del X, y_encoded

        rf_model = self._train_random_forest(X_train_split, y_train_split)
        rf_metrics = self._evaluate(rf_model, X_val_split, y_val_split, "RandomForest")

        xgb_model = self._train_xgboost(X_train_split, y_train_split)
        xgb_metrics = self._evaluate(xgb_model, X_val_split, y_val_split, "XGBoost")

        best_name, best_model, best_params, best_metrics = self._select_best(
            rf_model, rf_metrics, xgb_model, xgb_metrics
        )

        metadata = self._build_metadata(
            best_name=best_name,
            best_params=best_params,
            best_metrics=best_metrics,
            feature_names=feature_names,
            train_rows=len(X_train_split),
            val_rows=len(X_val_split),
        )

        self._save_artifacts(
            model=best_model,
            label_encoder=label_encoder,
            feature_names=feature_names,
            metadata=metadata,
        )

        elapsed = time.time() - start_time
        logger.info(
            "Training pipeline complete in %.1fs. Selected model: %s (weighted F1=%.4f).",
            elapsed,
            best_name,
            best_metrics["f1_weighted"],
        )
        return {"best_model_name": best_name, "metrics": best_metrics, "elapsed_seconds": elapsed}

    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Load ``X_train_fe.csv`` and ``y_train.csv`` exactly once each.

        Returns
        -------
        tuple of (pandas.DataFrame, pandas.Series, list of str)
            The feature matrix, the raw target series, and the ordered
            list of feature column names.

        Raises
        ------
        DataLoadError
            If either file is missing, empty, or row counts mismatch.
        """
        x_path = self.data_dir / "X_train_fe.csv"
        y_path = self.data_dir / "y_train.csv"

        if not x_path.is_file():
            raise DataLoadError(f"Required file not found: '{x_path}'.")
        if not y_path.is_file():
            raise DataLoadError(f"Required file not found: '{y_path}'.")

        try:
            X = pd.read_csv(x_path)
        except Exception as exc:
            raise DataLoadError(f"Failed to read '{x_path}': {exc}") from exc

        try:
            y_df = pd.read_csv(y_path)
        except Exception as exc:
            raise DataLoadError(f"Failed to read '{y_path}': {exc}") from exc

        if X.empty:
            raise DataLoadError(f"'{x_path}' contains no rows.")
        if y_df.empty:
            raise DataLoadError(f"'{y_path}' contains no rows.")
        if len(X) != len(y_df):
            raise DataLoadError(
                f"Row count mismatch between X_train_fe.csv ({len(X)}) and y_train.csv ({len(y_df)})."
            )
        if self._TARGET_COLUMN not in y_df.columns:
            raise DataLoadError(
                f"Target column '{self._TARGET_COLUMN}' not found in '{y_path}'. "
                f"Available columns: {list(y_df.columns)}"
            )

        # Downcast numeric feature columns to float32 in place to roughly
        # halve memory usage across ~2M rows without altering values
        # meaningfully for tree-based models.
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].astype(np.float32)

        feature_names = X.columns.tolist()
        y_series = y_df[self._TARGET_COLUMN]

        logger.info(
            "Loaded training data: X_train_fe.csv shape=%s, y_train.csv shape=%s.",
            X.shape,
            y_df.shape,
        )
        return X, y_series, feature_names

    def _encode_labels(self, y_raw: pd.Series) -> Tuple[LabelEncoder, np.ndarray]:
        """Fit a ``LabelEncoder`` on the raw attack-type labels.

        Parameters
        ----------
        y_raw : pandas.Series
            Raw string labels from the "Attack Type" column.

        Returns
        -------
        tuple of (sklearn.preprocessing.LabelEncoder, numpy.ndarray)
            The fitted encoder and the integer-encoded label array.

        Raises
        ------
        DataLoadError
            If encoding fails (e.g. due to unsupported label values).
        """
        try:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_raw)
        except Exception as exc:
            raise DataLoadError(f"Failed to encode target labels: {exc}") from exc

        logger.info("Encoded %d distinct attack-type classes.", len(label_encoder.classes_))
        return label_encoder, y_encoded

    def _split(
        self, X: pd.DataFrame, y_encoded: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Carve a stratified validation split out of the training data.

        This split exists solely to compare the two candidate models
        without touching ``X_test_fe.csv`` / ``y_test.csv``, which are
        reserved for ``evaluate_model.py``.

        Parameters
        ----------
        X : pandas.DataFrame
            Full engineered training feature matrix.
        y_encoded : numpy.ndarray
            Full integer-encoded target array, aligned with ``X``.

        Returns
        -------
        tuple
            ``(X_train_split, X_val_split, y_train_split, y_val_split)``.

        Raises
        ------
        TrainingError
            If the split cannot be performed (e.g. a class has too few
            samples to stratify).
        """
        try:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X,
                y_encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_encoded,
            )
        except Exception as exc:
            raise TrainingError(f"Failed to create train/validation split: {exc}") from exc

        logger.info(
            "Split data into %d training rows and %d validation rows.",
            len(X_train_split),
            len(X_val_split),
        )
        return X_train_split, X_val_split, y_train_split, y_val_split

    def _train_random_forest(self, X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
        """Fit exactly one ``RandomForestClassifier`` with fixed hyperparameters.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature matrix.
        y_train : numpy.ndarray
            Integer-encoded training labels.

        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            The fitted model.

        Raises
        ------
        TrainingError
            If fitting fails.
        """
        logger.info("Training RandomForestClassifier with params: %s", RANDOM_FOREST_PARAMS)
        try:
            model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
            model.fit(X_train, y_train)
        except Exception as exc:
            raise TrainingError(f"RandomForest training failed: {exc}") from exc
        logger.info("RandomForestClassifier training complete.")
        return model

    def _train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray) -> XGBClassifier:
        """Fit exactly one ``XGBClassifier`` with fixed hyperparameters.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature matrix.
        y_train : numpy.ndarray
            Integer-encoded training labels.

        Returns
        -------
        xgboost.XGBClassifier
            The fitted model.

        Raises
        ------
        TrainingError
            If fitting fails.
        """
        logger.info("Training XGBClassifier with params: %s", XGBOOST_PARAMS)
        try:
            num_classes = int(np.unique(y_train).size)
            model = XGBClassifier(num_class=num_classes, **XGBOOST_PARAMS)
            model.fit(X_train, y_train)
        except Exception as exc:
            raise TrainingError(f"XGBoost training failed: {exc}") from exc
        logger.info("XGBClassifier training complete.")
        return model

    def _evaluate(
        self, model: Any, X_val: pd.DataFrame, y_val: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """Score a fitted model on the validation split using weighted metrics.

        Parameters
        ----------
        model : estimator
            A fitted classifier exposing ``predict``.
        X_val : pandas.DataFrame
            Validation feature matrix.
        y_val : numpy.ndarray
            Integer-encoded validation labels.
        model_name : str
            Name used for logging.

        Returns
        -------
        dict
            Dictionary with keys ``accuracy``, ``precision_weighted``,
            ``recall_weighted``, and ``f1_weighted``.

        Raises
        ------
        TrainingError
            If prediction or metric computation fails.
        """
        try:
            y_pred = model.predict(X_val)
            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision_weighted": float(
                    precision_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
                "recall_weighted": float(
                    recall_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
                "f1_weighted": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
            }
        except Exception as exc:
            raise TrainingError(f"Evaluation of {model_name} failed: {exc}") from exc

        logger.info("%s validation metrics: %s", model_name, metrics)
        return metrics

    def _select_best(
        self,
        rf_model: RandomForestClassifier,
        rf_metrics: Dict[str, float],
        xgb_model: XGBClassifier,
        xgb_metrics: Dict[str, float],
    ) -> Tuple[str, Any, Dict[str, Any], Dict[str, float]]:
        """Select the better of the two candidates by weighted F1-score.

        Ties on weighted F1-score are broken by weighted precision.

        Parameters
        ----------
        rf_model : sklearn.ensemble.RandomForestClassifier
            Fitted Random Forest model.
        rf_metrics : dict
            Validation metrics for the Random Forest model.
        xgb_model : xgboost.XGBClassifier
            Fitted XGBoost model.
        xgb_metrics : dict
            Validation metrics for the XGBoost model.

        Returns
        -------
        tuple
            ``(best_name, best_model, best_hyperparameters, best_metrics)``.
        """
        rf_f1 = rf_metrics["f1_weighted"]
        xgb_f1 = xgb_metrics["f1_weighted"]

        if rf_f1 > xgb_f1:
            winner = ("RandomForest", rf_model, RANDOM_FOREST_PARAMS, rf_metrics)
        elif xgb_f1 > rf_f1:
            winner = ("XGBoost", xgb_model, XGBOOST_PARAMS, xgb_metrics)
        else:
            # Weighted F1 tie: fall back to weighted precision.
            if rf_metrics["precision_weighted"] >= xgb_metrics["precision_weighted"]:
                winner = ("RandomForest", rf_model, RANDOM_FOREST_PARAMS, rf_metrics)
            else:
                winner = ("XGBoost", xgb_model, XGBOOST_PARAMS, xgb_metrics)

        logger.info("Selected best model: %s", winner[0])
        return winner

    def _build_metadata(
        self,
        best_name: str,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, float],
        feature_names: list,
        train_rows: int,
        val_rows: int,
    ) -> Dict[str, Any]:
        """Assemble the training metadata dictionary.

        Parameters
        ----------
        best_name : str
            Name of the winning model ("RandomForest" or "XGBoost").
        best_params : dict
            Hyperparameters used to fit the winning model.
        best_metrics : dict
            Validation metrics of the winning model.
        feature_names : list of str
            Ordered feature names used for training.
        train_rows : int
            Number of rows used to fit the winning model.
        val_rows : int
            Number of rows used for validation-based model selection.

        Returns
        -------
        dict
            Metadata written to ``training_metadata.json``.
        """
        return {
            "best_model_name": best_name,
            "hyperparameters": best_params,
            "validation_metrics": best_metrics,
            "feature_count": len(feature_names),
            "training_rows": train_rows,
            "validation_rows": val_rows,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
        }

    def _save_artifacts(
        self,
        model: Any,
        label_encoder: LabelEncoder,
        feature_names: list,
        metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist the winning model and its supporting artifacts.

        Each artifact is first written to a temporary file in
        ``model_dir`` and only moved into its final name via
        ``os.replace`` after the write succeeds. If any step fails, no
        partially written or corrupted artifact is left behind under the
        final filenames.

        Parameters
        ----------
        model : estimator
            The winning fitted model.
        label_encoder : sklearn.preprocessing.LabelEncoder
            The fitted label encoder.
        feature_names : list of str
            Ordered feature names used for training.
        metadata : dict
            Training metadata to serialize as JSON.

        Raises
        ------
        ArtifactSaveError
            If any artifact fails to serialize or persist.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._atomic_joblib_dump(model, self.model_dir / "best_model.pkl")
            self._atomic_joblib_dump(label_encoder, self.model_dir / "label_encoder.pkl")
            self._atomic_write_csv(feature_names, self.model_dir / "selected_features.csv")
            self._atomic_write_json(metadata, self.model_dir / "training_metadata.json")
        except Exception as exc:
            raise ArtifactSaveError(f"Failed to persist model artifacts: {exc}") from exc

        logger.info("All model artifacts saved successfully to '%s'.", self.model_dir)

    @staticmethod
    def _atomic_joblib_dump(obj: Any, destination: Path) -> None:
        """Serialize ``obj`` via joblib to ``destination`` atomically."""
        fd, tmp_path = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump(obj, tmp_path)
            os.replace(tmp_path, destination)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_csv(feature_names: list, destination: Path) -> None:
        """Write the selected feature list to CSV atomically."""
        fd, tmp_path = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        os.close(fd)
        try:
            pd.DataFrame({"feature": feature_names}).to_csv(tmp_path, index=False)
            os.replace(tmp_path, destination)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_json(payload: Dict[str, Any], destination: Path) -> None:
        """Write a metadata dictionary to JSON atomically."""
        fd, tmp_path = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, indent=2)
            os.replace(tmp_path, destination)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()