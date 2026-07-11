"""
analytics.py
=============

Streamlit dashboard page for exploring analytics on the most recent
batch of predictions.

This module performs no training, inference, or feature engineering.
It only reads results already produced by the ``Prediction`` page and
stored in ``st.session_state``, and renders descriptive statistics and
visualizations over them.

Expected session state
-----------------------
This page expects the following keys to be populated in
``st.session_state`` by ``src/dashboard/prediction.py`` after a
successful prediction run:

    st.session_state["last_input_df"] : pandas.DataFrame
        The uploaded feature data used for the most recent prediction.
    st.session_state["last_predictions"] : pandas.DataFrame
        The corresponding prediction results, containing at least a
        ``prediction_label`` column and optionally a ``confidence``
        column.

If these keys are absent, this page prompts the user to run a
prediction first rather than raising an error.

Author: Member C
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Module-level configuration
# --------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

MAX_NUMERIC_FEATURES_DISPLAYED = 6


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def _get_session_results() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Retrieve the most recent prediction results from session state.

    Returns
    -------
    tuple[Optional[pandas.DataFrame], Optional[pandas.DataFrame]]
        A tuple of ``(input_df, predictions_df)``. Either element is
        ``None`` if not yet populated in ``st.session_state``.
    """
    input_df = st.session_state.get("last_input_df")
    predictions_df = st.session_state.get("last_predictions")
    return input_df, predictions_df


def _render_attack_severity_breakdown(predictions_df: pd.DataFrame) -> None:
    """
    Render a benign-vs-malicious breakdown of the prediction results.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Prediction results containing a ``prediction_label`` column.

    Returns
    -------
    None
    """
    st.subheader("⚖️ Benign vs. Malicious Breakdown")

    labels = predictions_df["prediction_label"].astype(str)
    is_benign = labels.str.upper().eq("BENIGN")

    benign_count = int(is_benign.sum())
    malicious_count = int((~is_benign).sum())
    total = benign_count + malicious_count

    if total == 0:
        st.info("No records available to summarize.")
        return

    breakdown_df = pd.DataFrame(
        {
            "Category": ["Benign", "Malicious"],
            "Count": [benign_count, malicious_count],
        }
    )

    col_chart, col_metrics = st.columns([2, 1])
    with col_chart:
        fig = px.pie(
            breakdown_df,
            names="Category",
            values="Count",
            title="Benign vs. Malicious Traffic",
            color="Category",
            color_discrete_map={"Benign": "#2ECC71", "Malicious": "#E74C3C"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.metric("Malicious Rate", f"{(malicious_count / total):.2%}")
        st.metric("Benign Records", f"{benign_count:,}")
        st.metric("Malicious Records", f"{malicious_count:,}")


def _render_confidence_distribution(predictions_df: pd.DataFrame) -> None:
    """
    Render a histogram of prediction confidence scores, if available.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Prediction results, optionally containing a ``confidence``
        column.

    Returns
    -------
    None
    """
    st.subheader("📉 Prediction Confidence Distribution")

    if "confidence" not in predictions_df.columns:
        st.info(
            "Confidence scores are not available for this batch of "
            "predictions (the underlying model may not support "
            "probability estimates)."
        )
        return

    fig = px.histogram(
        predictions_df,
        x="confidence",
        nbins=20,
        title="Distribution of Prediction Confidence",
        labels={"confidence": "Confidence"},
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    low_confidence_threshold = 0.6
    low_confidence_count = int((predictions_df["confidence"] < low_confidence_threshold).sum())
    if low_confidence_count > 0:
        st.warning(
            f"{low_confidence_count:,} record(s) were predicted with "
            f"confidence below {low_confidence_threshold:.0%} and may "
            "warrant manual review."
        )


def _render_feature_distributions(input_df: pd.DataFrame) -> None:
    """
    Render histograms for a sample of numeric input features.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The raw uploaded feature data used for prediction.

    Returns
    -------
    None
    """
    st.subheader("📐 Feature Distributions")

    numeric_columns = input_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.info("No numeric feature columns were found to visualize.")
        return

    default_selection = numeric_columns[:MAX_NUMERIC_FEATURES_DISPLAYED]
    selected_columns = st.multiselect(
        "Select feature(s) to visualize",
        options=numeric_columns,
        default=default_selection,
    )

    if not selected_columns:
        st.info("Select at least one feature to display its distribution.")
        return

    for column in selected_columns:
        fig = px.histogram(
            input_df, x=column, nbins=30, title=f"Distribution of '{column}'"
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_correlation_heatmap(input_df: pd.DataFrame) -> None:
    """
    Render a correlation heatmap across numeric input features.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The raw uploaded feature data used for prediction.

    Returns
    -------
    None
    """
    st.subheader("🔗 Feature Correlation Heatmap")

    numeric_df = input_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric feature columns are required to compute correlations.")
        return

    try:
        correlation_matrix = numeric_df.corr(numeric_only=True)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to compute correlation matrix.")
        st.error("Could not compute feature correlations for this dataset.")
        return

    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Streamlit page
# --------------------------------------------------------------------------

def render() -> None:
    """
    Render the 'Analytics' page of the Streamlit dashboard.

    This is the sole entry point intended to be called from
    ``src/dashboard/app.py``. It visualizes the most recent batch of
    predictions stored in ``st.session_state``, including a
    benign/malicious breakdown, confidence distribution, feature
    distributions, and a correlation heatmap.

    Returns
    -------
    None
    """
    st.title("📊 Analytics")
    st.markdown(
        "Explore deeper insights into the most recent batch of "
        "predictions, including confidence patterns and feature "
        "characteristics."
    )

    input_df, predictions_df = _get_session_results()

    if input_df is None or predictions_df is None:
        st.info(
            "No prediction results are available yet. Please run a "
            "prediction from the **Prediction** page first."
        )
        return

    if predictions_df.empty or input_df.empty:
        st.error("The stored prediction results are empty.")
        return

    if "prediction_label" not in predictions_df.columns:
        st.error(
            "Stored prediction results do not contain a "
            "'prediction_label' column and cannot be analyzed."
        )
        return

    st.divider()
    try:
        _render_attack_severity_breakdown(predictions_df)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render attack severity breakdown.")
        st.error("An error occurred while rendering the severity breakdown.")

    st.divider()
    try:
        _render_confidence_distribution(predictions_df)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render confidence distribution.")
        st.error("An error occurred while rendering the confidence distribution.")

    st.divider()
    try:
        _render_feature_distributions(input_df)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render feature distributions.")
        st.error("An error occurred while rendering feature distributions.")

    st.divider()
    try:
        _render_correlation_heatmap(input_df)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render correlation heatmap.")
        st.error("An error occurred while rendering the correlation heatmap.")


if __name__ == "__main__":
    render()