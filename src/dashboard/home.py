"""
home.py
========

Streamlit dashboard landing page for the Enterprise Network Intrusion
Intelligence System.

This module renders the overview/home page: a project summary, a
high-level explanation of the detection workflow, a quick-start guide,
and (when available) a snapshot of the currently deployed model's
training metadata.

This module performs no training, inference, or feature engineering. It
only reads ``models/training_metadata.json`` for display purposes and
never loads, fits, or modifies any model artifact.

Author: Member C
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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

METADATA_PATH = Path("models") / "training_metadata.json"


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_training_metadata(path: str = str(METADATA_PATH)) -> Optional[Dict[str, Any]]:
    """
    Load training metadata for display on the home page.

    Parameters
    ----------
    path : str, optional
        Path to ``training_metadata.json``. Defaults to the standard
        project location under ``models/``.

    Returns
    -------
    Optional[dict]
        Parsed metadata dictionary, or ``None`` if the file is missing
        or unreadable. A missing metadata file is not treated as a
        fatal error since the home page can render without it.
    """
    metadata_path = Path(path)
    if not metadata_path.is_file():
        logger.warning("Training metadata file not found at '%s'.", metadata_path)
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        logger.info("Training metadata loaded successfully from '%s'.", metadata_path)
        return metadata
    except json.JSONDecodeError as exc:
        logger.error("Training metadata file is not valid JSON: %s", exc)
        return None
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while loading training metadata.")
        return None


def _render(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Render a summary card of the currently deployed model, if metadata exists.

    Parameters
    ----------
    metadata : Optional[dict]
        Parsed training metadata, as returned by
        :func:`_load_training_metadata`. If ``None``, an informational
        placeholder is shown instead.

    Returns
    -------
    None
    """
    st.subheader("🧠 Deployed Model Snapshot")

    if metadata is None:
        st.info(
            "Model metadata is not currently available. Once "
            "`models/training_metadata.json` is present, a summary of the "
            "deployed model will appear here."
        )
        return

    model_name = metadata.get("best_model_name", "Unknown")

    validation_metrics = metadata.get("validation_metrics", {})

    accuracy = validation_metrics.get("accuracy")
    f1_score = validation_metrics.get("f1_weighted")

    training_date = metadata.get("trained_at")

    snapshot_cols = st.columns(4)
    snapshot_cols[0].metric("Model Type", str(model_name))
    snapshot_cols[1].metric(
        "Test Accuracy", f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "N/A"
    )
    snapshot_cols[2].metric(
        "F1 Score", f"{f1_score:.3f}" if isinstance(f1_score, (int, float)) else "N/A"
    )
    snapshot_cols[3].metric("Trained On", str(training_date) if training_date else "N/A")


# --------------------------------------------------------------------------
# Streamlit page
# --------------------------------------------------------------------------

def render() -> None:
    """
    Render the 'Home' page of the Streamlit dashboard.

    This is the sole entry point intended to be called from
    ``src/dashboard/app.py``. It presents the project overview, the
    detection workflow, a quick-start guide, and a model snapshot.

    Returns
    -------
    None
    """
    st.title("🛡️ Enterprise Network Intrusion Intelligence System")
    st.markdown(
        "A machine learning-powered platform for detecting and "
        "classifying network intrusions, built on the **CICIDS2017** "
        "dataset."
    )

    st.divider()

    st.subheader("📖 About This System")
    st.markdown(
        "This system classifies network traffic flows as benign or as "
        "one of several known attack categories (e.g. DoS, DDoS, "
        "port scanning, brute force, and web attacks). Records are "
        "scored using a model trained on labeled traffic features "
        "extracted from real network captures.\n\n"
        "The dashboard is organized into the following sections:"
    )

    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown("**🔍 Prediction**")
        st.caption(
            "Upload a CSV of traffic records and classify each one, with "
            "confidence scores and attack-type breakdowns."
        )
    with feature_cols[1]:
        st.markdown("**📊 Analytics**")
        st.caption(
            "Explore trends and distributions across prediction history "
            "and dataset characteristics."
        )
    with feature_cols[2]:
        st.markdown("**ℹ️ About**")
        st.caption(
            "Learn about the dataset, modeling approach, and project team "
            "behind this system."
        )

    st.divider()

    metadata = _load_training_metadata()
    _render(metadata)

    st.divider()

    st.subheader("🚀 Quick Start")
    st.markdown(
        "1. Navigate to the **Prediction** page.\n"
        "2. Upload a CSV file containing network flow records.\n"
        "3. Click **Predict** to classify all records.\n"
        "4. Review the summary, charts, and prediction table.\n"
        "5. Download the results as a CSV file."
    )

    st.divider()
    st.caption(
        "Enterprise Network Intrusion Intelligence System · "
        "Powered by CICIDS2017 · Academic project"
    )


if __name__ == "__main__":
    render()