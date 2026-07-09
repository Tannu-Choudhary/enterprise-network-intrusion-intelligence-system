"""
prediction.py
==============

Prediction page for the Enterprise Network Intrusion Intelligence System
dashboard.

This module allows an end user to classify network traffic flows as
benign or malicious (and by attack sub-type) using the trained model
artifacts produced by the modeling pipeline:

    - models/best_model.pkl
    - models/scaler.pkl
    - models/label_encoder.pkl
    - models/selected_features.csv

Two input modes are supported:
    1. **CSV Upload** — batch prediction over one or more flow records.
    2. **Manual Entry** — single-record prediction via dynamically
       generated input widgets based on the selected feature list.

Wherever possible, this module delegates the actual inference logic to
``src.inference.predictor`` (owned by Member B) to avoid duplicating
prediction logic. If that module is not yet available, a local fallback
implementation is used so this page remains independently testable.

Author: Member C
Python: 3.11
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.utils.logger import get_logger
except ImportError:  # pragma: no cover - fallback until utils/logger.py exists
    def get_logger(name: str) -> logging.Logger:
        """Fallback logger factory mirroring the shared logging interface."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

try:
    from src.utils.config import MODELS_DIR
except ImportError:  # pragma: no cover - fallback until utils/config.py exists
    MODELS_DIR = Path("models")

logger = get_logger(__name__)

MODEL_PATH = Path(MODELS_DIR) / "best_model.pkl"
SCALER_PATH = Path(MODELS_DIR) / "scaler.pkl"
LABEL_ENCODER_PATH = Path(MODELS_DIR) / "label_encoder.pkl"
SELECTED_FEATURES_PATH = Path(MODELS_DIR) / "selected_features.csv"

BENIGN_LABEL = "BENIGN"


# --------------------------------------------------------------------------
# Optional delegation to Member B's inference module
# --------------------------------------------------------------------------
try:
    from src.inference.predictor import predict_batch as _external_predict_batch
    _HAS_EXTERNAL_PREDICTOR = True
    logger.info("Using external predictor from src.inference.predictor.")
except ImportError:  # pragma: no cover - fallback until predictor.py exists
    _external_predict_batch = None
    _HAS_EXTERNAL_PREDICTOR = False
    logger.info(
        "src.inference.predictor not found; using local fallback prediction "
        "logic in prediction.py."
    )


# --------------------------------------------------------------------------
# Cached artifact loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def _load_artifacts() -> Optional[Tuple[Any, Any, Any, List[str]]]:
    """
    Load the trained model, scaler, label encoder, and selected feature
    list from disk.

    Results are cached for the lifetime of the Streamlit session via
    ``st.cache_resource`` to avoid re-reading pickle files on every
    interaction.

    Returns:
        A tuple of (model, scaler, label_encoder, selected_features) if
        all artifacts load successfully, otherwise None.
    """
    try:
        for path in (MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, SELECTED_FEATURES_PATH):
            if not path.exists():
                logger.warning("Missing model artifact: %s", path)
                return None

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        features_df = pd.read_csv(SELECTED_FEATURES_PATH)

        feature_column = features_df.columns[0]
        selected_features = features_df[feature_column].astype(str).tolist()

        logger.info(
            "Loaded model artifacts successfully (%d selected features).",
            len(selected_features),
        )
        return model, scaler, label_encoder, selected_features

    except (OSError, ValueError) as exc:
        logger.exception("Failed to load model artifacts: %s", exc)
        return None
    except Exception:  # noqa: BLE001 - broad safety net around joblib.load
        logger.exception("Unexpected error while loading model artifacts.")
        return None


def _predict_local_fallback(
    dataframe: pd.DataFrame,
    model: Any,
    scaler: Any,
    label_encoder: Any,
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Local fallback prediction routine, used only if
    ``src.inference.predictor`` is not yet available.

    Args:
        dataframe: Input records containing at least the selected
            feature columns.
        model: Trained classifier exposing a ``predict`` method (and
            optionally ``predict_proba``).
        scaler: Fitted scaler exposing a ``transform`` method.
        label_encoder: Fitted label encoder exposing
            ``inverse_transform``.
        selected_features: Ordered list of feature column names
            expected by the model.

    Returns:
        A copy of the relevant rows with two additional columns:
        'predicted_label' and 'confidence' (max class probability, if
        available).

    Raises:
        KeyError: If required feature columns are missing from
            ``dataframe``.
    """
    missing_columns = [col for col in selected_features if col not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"Missing required feature columns: {missing_columns}")

    feature_matrix = dataframe[selected_features].to_numpy(dtype=float)
    scaled_matrix = scaler.transform(feature_matrix)

    predictions = model.predict(scaled_matrix)
    decoded_labels = label_encoder.inverse_transform(predictions)

    confidences: np.ndarray
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_matrix)
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = np.full(shape=(len(predictions),), fill_value=np.nan)

    result = dataframe.copy()
    result["predicted_label"] = decoded_labels
    result["confidence"] = confidences
    return result


def _run_prediction(dataframe: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Execute prediction over the given dataframe, delegating to the
    external predictor module when available, otherwise using the
    local fallback.

    Args:
        dataframe: Input records to classify.

    Returns:
        A dataframe with prediction results appended, or None if
        prediction could not be performed (e.g. missing artifacts).
    """
    artifacts = _load_artifacts()
    if artifacts is None:
        st.error(
            "Model artifacts not found in `models/`. Please ensure "
            "`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and "
            "`selected_features.csv` have been generated by the training "
            "pipeline."
        )
        return None

    model, scaler, label_encoder, selected_features = artifacts

    try:
        if _HAS_EXTERNAL_PREDICTOR and _external_predict_batch is not None:
            logger.info("Delegating prediction to src.inference.predictor.")
            result = _external_predict_batch(dataframe)
        else:
            result = _predict_local_fallback(
                dataframe, model, scaler, label_encoder, selected_features
            )
        logger.info("Prediction completed for %d record(s).", len(result))
        return result

    except KeyError as exc:
        logger.error("Feature mismatch during prediction: %s", exc)
        st.error(f"Input data is missing required columns: {exc}")
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during prediction.")
        st.error("An unexpected error occurred while running the prediction.")
        return None


def _render_results(result_df: pd.DataFrame) -> None:
    """
    Render prediction results with summary highlights and a detail table.

    Args:
        result_df: Dataframe containing a 'predicted_label' column (and
            optionally 'confidence').
    """
    try:
        total = len(result_df)
        malicious_count = int((result_df["predicted_label"] != BENIGN_LABEL).sum())
        benign_count = total - malicious_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{total:,}")
        col2.metric("Benign", f"{benign_count:,}")
        col3.metric("Malicious", f"{malicious_count:,}", delta_color="inverse")

        if malicious_count > 0:
            st.error(f"⚠️ {malicious_count} malicious flow(s) detected!")
        else:
            st.success("✅ No malicious traffic detected in this batch.")

        st.subheader("Detailed Results")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_bytes,
            file_name="prediction_results.csv",
            mime="text/csv",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render prediction results.")
        st.error("Unable to display prediction results.")


def _render_csv_upload_mode(selected_features: List[str]) -> None:
    """
    Render the CSV upload input mode for batch prediction.

    Args:
        selected_features: Feature columns expected by the model, shown
            to the user as guidance.
    """
    st.caption(
        "Upload a CSV file containing network flow records. The file "
        f"must include the following {len(selected_features)} feature "
        "column(s): " + ", ".join(selected_features[:10])
        + (", ..." if len(selected_features) > 10 else "")
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        raw_bytes = uploaded_file.getvalue()
        dataframe = pd.read_csv(io.BytesIO(raw_bytes))
        st.write(f"Loaded {len(dataframe):,} record(s) from `{uploaded_file.name}`.")

        if st.button("Run Prediction", key="predict_csv"):
            with st.spinner("Running inference..."):
                result_df = _run_prediction(dataframe)
            if result_df is not None:
                _render_results(result_df)

    except pd.errors.ParserError as exc:
        logger.error("Failed to parse uploaded CSV: %s", exc)
        st.error("The uploaded file could not be parsed as a valid CSV.")
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while handling uploaded CSV.")
        st.error("An unexpected error occurred while processing the file.")


def _render_manual_entry_mode(selected_features: List[str]) -> None:
    """
    Render the manual entry input mode for single-record prediction.

    Dynamically generates a numeric input widget for each selected
    feature. Intended for quick, ad-hoc testing rather than bulk use.

    Args:
        selected_features: Feature columns expected by the model.
    """
    st.caption(
        f"Enter values for all {len(selected_features)} required feature(s) "
        "below, then click Predict."
    )

    with st.form(key="manual_entry_form"):
        input_values = {}
        num_columns = 3
        columns = st.columns(num_columns)
        for index, feature_name in enumerate(selected_features):
            target_column = columns[index % num_columns]
            input_values[feature_name] = target_column.number_input(
                label=feature_name,
                value=0.0,
                format="%.6f",
                key=f"manual_{feature_name}",
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            single_record_df = pd.DataFrame([input_values])
            with st.spinner("Running inference..."):
                result_df = _run_prediction(single_record_df)
            if result_df is not None:
                _render_results(result_df)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error during manual entry prediction.")
            st.error("An unexpected error occurred while running the prediction.")


def render() -> None:
    """
    Render the Prediction page.

    This is the public entry point invoked by ``src/dashboard/app.py``.
    All exceptions are caught locally so that a failure on this page
    does not propagate and crash the rest of the dashboard.
    """
    try:
        st.title("🔍 Network Traffic Prediction")
        st.markdown(
            "Classify network flow records as **benign** or **malicious** "
            "using the trained intrusion detection model."
        )

        artifacts = _load_artifacts()
        if artifacts is None:
            st.warning(
                "Model artifacts are not yet available in `models/`. "
                "This page will remain limited until the training "
                "pipeline has produced `best_model.pkl`, `scaler.pkl`, "
                "`label_encoder.pkl`, and `selected_features.csv`.",
                icon="⚠️",
            )
            return

        _, _, _, selected_features = artifacts

        input_mode = st.radio(
            "Input mode",
            options=["Upload CSV (batch)", "Manual entry (single record)"],
            horizontal=True,
        )

        st.markdown("---")

        if input_mode == "Upload CSV (batch)":
            _render_csv_upload_mode(selected_features)
        else:
            _render_manual_entry_mode(selected_features)

        logger.info("Prediction page rendered successfully.")

    except Exception:  # noqa: BLE001 - page-level safety net
        logger.exception("Unhandled exception while rendering the Prediction page.")
        st.error(
            "An unexpected error occurred while loading the Prediction page. "
            "Please check the application logs for details."
        )