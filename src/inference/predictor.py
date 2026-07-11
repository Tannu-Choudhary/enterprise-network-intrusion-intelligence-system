"""Inference module for the Enterprise Network Intrusion Intelligence System.

This module loads the persisted best model artifact (Random Forest or
XGBoost, whichever was selected during training) together with its
supporting artifacts -- the label encoder, the selected feature list, and
the training metadata -- and exposes a ``Predictor`` class capable of
scoring new, unseen network flow records.

The module is intentionally decoupled from the training pipeline: it never
re-fits, re-selects features, or mutates any of the loaded artifacts. It
only performs feature alignment and inference.

Examples
--------
>>> from src.inference.predictor import Predictor
>>> predictor = Predictor()
>>> predictions = predictor.predict(X_new)
>>> probabilities = predictor.predict_proba(X_new)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictorError(Exception):
    """Base exception for all predictor-related failures."""


class ModelArtifactsNotFoundError(PredictorError):
    """Raised when one or more required model artifacts cannot be located.

    Parameters
    ----------
    message : str
        Human-readable description of which artifact is missing.
    """


class FeatureMismatchError(PredictorError):
    """Raised when input data does not contain the required feature set.

    Parameters
    ----------
    message : str
        Human-readable description of the missing or mismatched columns.
    """


class Predictor:
    """Loads persisted model artifacts and serves intrusion predictions.

    On instantiation, the predictor loads the best trained model, the
    label encoder, the ordered list of selected features, and the training
    metadata exactly once. Every subsequent call to :meth:`predict` or
    :meth:`predict_proba` reuses these in-memory artifacts.

    Parameters
    ----------
    model_dir : str or pathlib.Path, optional
        Directory containing ``best_model.pkl``, ``label_encoder.pkl``,
        and ``selected_features.csv``. If ``None``, the value is taken
        from ``Config.MODELS_DIR`` (falling back to ``"models"``).

    Attributes
    ----------
    model_dir : pathlib.Path
        Resolved directory holding the model artifacts.
    selected_features : list of str
        Ordered feature names expected by the loaded model.
    metadata : dict
        Contents of ``training_metadata.json``, including the name of the
        selected model and its recorded evaluation metrics.

    Raises
    ------
    ModelArtifactsNotFoundError
        If any required artifact file is missing from ``model_dir``.
    PredictorError
        If an artifact exists but cannot be deserialized.
    """

    _REQUIRED_ARTIFACTS = (
        "best_model.pkl",
        "label_encoder.pkl",
        "selected_features.csv",
    )

    def __init__(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        self.model_dir: Path = Path(
            model_dir if model_dir is not None else getattr(get_config(), "MODELS_DIR", "models")
        )
        self._model: Any = None
        self._label_encoder: Any = None
        self.selected_features: List[str] = []
        self.metadata: Dict[str, Any] = {}

        logger.info("Initializing Predictor from artifacts in '%s'.", self.model_dir)
        self._check_artifacts_exist()
        self._load_selected_features()
        self._load_model()
        self._load_label_encoder()
        self._load_metadata()
        logger.info(
            "Predictor ready. Model type: %s | Features expected: %d",
            self.metadata.get("best_model_name", type(self._model).__name__),
            len(self.selected_features),
        )

    def _check_artifacts_exist(self) -> None:
        """Verify that all mandatory artifact files exist on disk.

        Raises
        ------
        ModelArtifactsNotFoundError
            If ``model_dir`` itself or any required file is missing.
        """
        if not self.model_dir.exists():
            message = f"Model directory not found: '{self.model_dir}'."
            logger.error(message)
            raise ModelArtifactsNotFoundError(message)

        missing = [
            name
            for name in self._REQUIRED_ARTIFACTS
            if not (self.model_dir / name).is_file()
        ]
        if missing:
            message = (
                f"Missing required model artifact(s) in '{self.model_dir}': "
                f"{missing}. Ensure train_model.py has completed successfully."
            )
            logger.error(message)
            raise ModelArtifactsNotFoundError(message)

    def _load_selected_features(self) -> None:
        """Load the ordered list of selected feature names.

        Raises
        ------
        PredictorError
            If the CSV cannot be read or does not contain a feature column.
        """
        path = self.model_dir / "selected_features.csv"
        try:
            features_df = pd.read_csv(path)
        except Exception as exc:
            message = f"Failed to read selected features file '{path}': {exc}"
            logger.error(message)
            raise PredictorError(message) from exc

        if features_df.empty:
            message = f"Selected features file '{path}' is empty."
            logger.error(message)
            raise PredictorError(message)

        # The feature name column may be the first column regardless of
        # its exact header (e.g. "feature", "feature_name").
        column_name = features_df.columns[0]
        self.selected_features = features_df[column_name].astype(str).tolist()
        logger.debug("Loaded %d selected feature names.", len(self.selected_features))

    def _load_model(self) -> None:
        """Deserialize the best trained model.

        Raises
        ------
        PredictorError
            If the model file exists but cannot be deserialized, or is
            empty/corrupted.
        """
        path = self.model_dir / "best_model.pkl"
        try:
            if path.stat().st_size == 0:
                raise PredictorError(f"Model file '{path}' is empty or corrupted.")
            self._model = joblib.load(path)
        except PredictorError:
            raise
        except Exception as exc:
            message = f"Failed to load model from '{path}': {exc}"
            logger.error(message)
            raise PredictorError(message) from exc

        if not hasattr(self._model, "predict"):
            message = f"Deserialized object from '{path}' is not a valid estimator."
            logger.error(message)
            raise PredictorError(message)

    def _load_label_encoder(self) -> None:
        """Deserialize the label encoder used to decode class predictions.

        Raises
        ------
        PredictorError
            If the label encoder cannot be deserialized.
        """
        path = self.model_dir / "label_encoder.pkl"
        try:
            self._label_encoder = joblib.load(path)
        except Exception as exc:
            message = f"Failed to load label encoder from '{path}': {exc}"
            logger.error(message)
            raise PredictorError(message) from exc

    def _load_metadata(self) -> None:
        """Load training metadata if available.

        Metadata is optional for inference itself, so a missing or
        unreadable file only logs a warning rather than raising.
        """
        path = self.model_dir / "training_metadata.json"
        if not path.is_file():
            logger.warning("Training metadata file not found at '%s'.", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                self.metadata = json.load(file_handle)
        except Exception as exc:
            logger.warning("Failed to parse training metadata '%s': %s", path, exc)
            self.metadata = {}

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and reorder input columns to match the trained feature set.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw input records. Must contain, at minimum, every column in
            ``self.selected_features``. Extra columns are ignored.

        Returns
        -------
        pandas.DataFrame
            View of ``X`` restricted and reordered to ``self.selected_features``.

        Raises
        ------
        FeatureMismatchError
            If one or more required feature columns are absent from ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            message = f"Expected a pandas DataFrame, got {type(X).__name__}."
            logger.error(message)
            raise FeatureMismatchError(message)

        missing_columns = [col for col in self.selected_features if col not in X.columns]
        if missing_columns:
            message = (
                f"Input data is missing {len(missing_columns)} required feature "
                f"column(s): {missing_columns}"
            )
            logger.error(message)
            raise FeatureMismatchError(message)

        # Select without copying the full frame; only the required columns
        # are materialized, in the exact order the model expects.
        return X.loc[:, self.selected_features]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict attack type labels for new network flow records.

        Parameters
        ----------
        X : pandas.DataFrame
            Input records containing at least all columns listed in
            ``self.selected_features``.

        Returns
        -------
        numpy.ndarray
            Array of decoded string labels (e.g. ``"BENIGN"``, ``"DDoS"``)
            with one entry per input row.

        Raises
        ------
        FeatureMismatchError
            If required feature columns are missing from ``X``.
        PredictorError
            If inference fails for any other reason.
        """
        aligned = self._align_features(X)
        try:
            encoded_predictions = self._model.predict(aligned)
            decoded_predictions = self._label_encoder.inverse_transform(encoded_predictions)
        except Exception as exc:
            message = f"Model inference failed during predict(): {exc}"
            logger.error(message)
            raise PredictorError(message) from exc

        logger.info("Generated %d predictions.", len(decoded_predictions))
        return decoded_predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for new network flow records.

        Parameters
        ----------
        X : pandas.DataFrame
            Input records containing at least all columns listed in
            ``self.selected_features``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, n_classes)`` with class
            probabilities. Column order matches
            ``self.get_class_labels()``.

        Raises
        ------
        FeatureMismatchError
            If required feature columns are missing from ``X``.
        PredictorError
            If the underlying model does not support probability
            estimates, or inference fails for any other reason.
        """
        if not hasattr(self._model, "predict_proba"):
            message = f"Loaded model '{type(self._model).__name__}' does not support predict_proba()."
            logger.error(message)
            raise PredictorError(message)

        aligned = self._align_features(X)
        try:
            probabilities = self._model.predict_proba(aligned)
        except Exception as exc:
            message = f"Model inference failed during predict_proba(): {exc}"
            logger.error(message)
            raise PredictorError(message) from exc

        logger.info("Generated probability estimates for %d records.", len(probabilities))
        return probabilities

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the attack type for a single network flow record.

        Convenience wrapper around :meth:`predict` and :meth:`predict_proba`
        intended for dashboard or API use cases with one record at a time.

        Parameters
        ----------
        record : dict
            Mapping of feature name to value. Must contain, at minimum,
            every column in ``self.selected_features``.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``"prediction"`` : str, the decoded predicted label.
            - ``"confidence"`` : float or None, the probability of the
              predicted class if the model supports ``predict_proba``,
              otherwise ``None``.
            - ``"class_probabilities"`` : dict or None, mapping of every
              class label to its predicted probability.

        Raises
        ------
        FeatureMismatchError
            If required feature columns are missing from ``record``.
        PredictorError
            If inference fails for any other reason.
        """
        single_row = pd.DataFrame([record])
        prediction = self.predict(single_row)[0]

        result: Dict[str, Any] = {
            "prediction": prediction,
            "confidence": None,
            "class_probabilities": None,
        }

        if hasattr(self._model, "predict_proba"):
            try:
                probabilities = self.predict_proba(single_row)[0]
                class_labels = self.get_class_labels()
                class_probabilities = dict(zip(class_labels, probabilities.tolist()))
                result["class_probabilities"] = class_probabilities
                result["confidence"] = float(np.max(probabilities))
            except PredictorError:
                logger.warning("Could not compute probabilities for single-record prediction.")

        return result

    def get_class_labels(self) -> List[str]:
        """Return the list of class labels known to the label encoder.

        Returns
        -------
        list of str
            Class labels in the order used internally by the model
            (matches the column order of ``predict_proba`` output).
        """
        return list(self._label_encoder.classes_)

    @property
    def model_info(self) -> Dict[str, Any]:
        """dict: Training metadata for the currently loaded model."""
        return self.metadata