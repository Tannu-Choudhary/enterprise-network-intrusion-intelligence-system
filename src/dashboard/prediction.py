"""
prediction.py
==============

Streamlit dashboard page for running network intrusion predictions on
user-uploaded CSV files.

This module is strictly a *consumer* of the existing inference pipeline
defined in ``src/inference/predictor.py``. It performs no model training,
no feature engineering, and no artifact fitting of any kind. Its sole
responsibilities are:

    1. Accept a CSV upload from the user via the Streamlit UI.
    2. Delegate feature validation and prediction to the existing
       ``Predictor`` class.
    3. Render prediction results, summary statistics, visualizations,
       and a downloadable results file.

This module relies on ``src.inference.predictor.Predictor``, which
already loads ``best_model.pkl``, ``label_encoder.pkl``, and
``selected_features.csv`` and exposes ``predict()``, ``predict_proba()``,
and ``get_class_labels()``. No model artifacts are loaded directly here.

Author: Member B
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference.predictor import (
    FeatureMismatchError,
    ModelArtifactsNotFoundError,
    Predictor,
    PredictorError,
)

# --------------------------------------------------------------------------
# Module-level configuration
# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------
# Helper / caching functions
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_predictor() -> Optional[Predictor]:
    """
    Instantiate and cache the ``Predictor``.

    Returns
    -------
    Optional[Predictor]
        A ready-to-use ``Predictor`` instance, or ``None`` if the model
        artifacts could not be loaded.

    Notes
    -----
    This function does not fit, train, or modify any artifact. It only
    instantiates the pre-existing ``Predictor`` class, which loads
    ``best_model.pkl``, ``label_encoder.pkl``, and
    ``selected_features.csv`` internally. The instance is cached for the
    lifetime of the Streamlit session to avoid reloading the model on
    every rerun.
    """
    try:
        predictor = Predictor()
        logger.info("Predictor initialized successfully.")
        return predictor
    except ModelArtifactsNotFoundError as exc:
        logger.error("Model artifacts not found: %s", exc)
        return None
    except PredictorError as exc:
        logger.error("Failed to initialize Predictor: %s", exc)
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while initializing Predictor.")
        return None


def _validate_uploaded_csv(
    df: pd.DataFrame, required_features: list
) -> tuple[bool, list]:
    """
    Pre-validate that the uploaded DataFrame contains all required features.

    This is a UX convenience check performed before calling
    ``Predictor.predict()``, which enforces the same requirement
    internally via ``FeatureMismatchError``.

    Parameters
    ----------
    df : pandas.DataFrame
        The uploaded, user-provided data.
    required_features : list
        Feature columns required by the model
        (``Predictor.selected_features``).

    Returns
    -------
    tuple[bool, list]
        A tuple of ``(is_valid, missing_columns)``.
    """
    missing = [col for col in required_features if col not in df.columns]
    return (len(missing) == 0, missing)


def _run_prediction(
    df: pd.DataFrame, predictor: Predictor
) -> Optional[pd.DataFrame]:
    """
    Run predictions via the existing ``Predictor`` instance.

    Parameters
    ----------
    df : pandas.DataFrame
        Uploaded input data. May contain extra columns; ``Predictor``
        selects and reorders only the required feature columns.
    predictor : Predictor
        A loaded, ready-to-use predictor instance.

    Returns
    -------
    Optional[pandas.DataFrame]
        DataFrame with a ``prediction_label`` column and, when the
        underlying model supports probability estimates, a
        ``confidence`` column holding the top-class probability per
        record. Returns ``None`` if prediction failed.
    """
    try:
        labels = predictor.predict(df)
    except FeatureMismatchError as exc:
        logger.error("Feature mismatch during prediction: %s", exc)
        st.error(f"Prediction failed due to a feature mismatch: {exc}")
        return None
    except PredictorError as exc:
        logger.error("Prediction failed: %s", exc)
        st.error(f"Prediction failed: {exc}")
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during prediction.")
        st.error("An unexpected error occurred while generating predictions.")
        return None

    results_df = pd.DataFrame({"prediction_label": labels})

    try:
        probabilities = predictor.predict_proba(df)
        results_df["confidence"] = probabilities.max(axis=1)
        logger.info("Confidence scores computed for %d records.", len(results_df))
    except PredictorError as exc:
        # Model does not support predict_proba, or probability inference
        # failed. Confidence is optional, so this is not fatal.
        logger.warning("Confidence scores unavailable: %s", exc)

    return results_df


# --------------------------------------------------------------------------
# Streamlit page
# --------------------------------------------------------------------------

def render() -> None:
    """
    Render the 'Prediction' page of the Streamlit dashboard.

    This is the sole entry point intended to be called from
    ``src/dashboard/app.py``. It orchestrates file upload, validation,
    inference, results display, visualizations, and CSV export.

    Returns
    -------
    None
    """
    st.title("🔍 Network Intrusion Prediction")
    st.markdown(
        "Upload network traffic data as a CSV file to classify each "
        "record using the trained intrusion detection model."
    )

    predictor = _load_predictor()
    if predictor is None:
        st.error(
            "Model artifacts could not be loaded. Please verify that "
            "`best_model.pkl`, `label_encoder.pkl`, and "
            "`selected_features.csv` exist under the `models/` directory."
        )
        return

    model_name = predictor.model_info.get("best_model_name")
    if model_name:
        st.caption(f"Active model: **{model_name}**")

    uploaded_file = st.file_uploader(
        "Upload traffic data (CSV format)", type=["csv"]
    )

    if uploaded_file is None:
        st.info("Awaiting CSV upload.")
        return

    try:
        input_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
        return
    except pd.errors.ParserError:
        st.error("The uploaded file could not be parsed. Please upload a valid CSV.")
        return
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while reading the uploaded CSV.")
        st.error("An unexpected error occurred while reading the uploaded file.")
        return

    if input_df.empty:
        st.error("The uploaded CSV file contains no rows.")
        return

    is_valid, missing_columns = _validate_uploaded_csv(
        input_df, predictor.selected_features
    )
    if not is_valid:
        st.error(
            "The uploaded CSV is missing required feature column(s): "
            f"{', '.join(missing_columns)}"
        )
        return

    st.success(f"File uploaded successfully — {len(input_df):,} records found.")
    st.dataframe(input_df.head(10), use_container_width=True)

    if not st.button("Predict", type="primary"):
        return

    with st.spinner("Running predictions..."):
        results_df = _run_prediction(input_df, predictor)

    if results_df is None:
        return
# Save results for Analytics page
    st.session_state["last_input_df"] = input_df.copy()
    st.session_state["last_predictions"] = results_df.copy()

    logger.info("Prediction results stored in Streamlit session state.")    
    st.divider()
    st.subheader("📊 Prediction Summary")

    has_confidence = "confidence" in results_df.columns
    total_records = len(results_df)
    attack_counts = results_df["prediction_label"].value_counts()
    top_prediction = attack_counts.idxmax()

    summary_cols = st.columns(3)
    summary_cols[0].metric("Total Records", f"{total_records:,}")
    summary_cols[1].metric("Most Frequent Prediction", str(top_prediction))
    if has_confidence:
        avg_confidence = results_df["confidence"].mean()
        summary_cols[2].metric("Avg. Confidence", f"{avg_confidence:.2%}")
    else:
        summary_cols[2].metric("Avg. Confidence", "N/A")

    st.divider()
    st.subheader("📈 Attack Type Distribution")

    chart_cols = st.columns(2)

    distribution_df = attack_counts.reset_index()
    distribution_df.columns = ["Attack Type", "Count"]

    with chart_cols[0]:
        bar_fig = px.bar(
            distribution_df,
            x="Attack Type",
            y="Count",
            color="Attack Type",
            title="Attack Type Distribution",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with chart_cols[1]:
        pie_fig = px.pie(
            distribution_df,
            names="Attack Type",
            values="Count",
            title="Attack Type Percentage",
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Prediction Table")

    display_columns = ["prediction_label"] + (["confidence"] if has_confidence else [])
    combined_df = pd.concat(
        [input_df.reset_index(drop=True), results_df[display_columns]], axis=1
    )
    st.dataframe(combined_df, use_container_width=True)

    st.divider()
    csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_bytes,
        file_name="intrusion_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    render()