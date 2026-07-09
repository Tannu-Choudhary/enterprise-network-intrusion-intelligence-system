"""
home.py
=======

Landing page for the Enterprise Network Intrusion Intelligence System
dashboard.

This module renders the "Home" tab, which provides:
    - A high-level introduction to the project and its objectives.
    - A summary of the CICIDS2017 dataset used to train the underlying
      intrusion-detection model.
    - Key at-a-glance metrics (if pre-computed metadata is available).
    - Quick-navigation guidance pointing users to the Prediction and
      Analytics pages.

This module is intentionally read-only / informational: it does not
perform inference or heavy data processing. Any expensive computation
should live in src/models or src/features (owned by other members) and
be exposed here only as lightweight, cached summaries.

Author: Member C
Python: 3.11
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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
    from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR
except ImportError:  # pragma: no cover - fallback until utils/config.py exists
    PROCESSED_DATA_DIR = Path("data/processed")
    REPORTS_DIR = Path("reports")


logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Static project metadata
# --------------------------------------------------------------------------
ATTACK_CATEGORIES = [
    "BENIGN",
    "DoS / DDoS",
    "PortScan",
    "Brute Force (FTP/SSH)",
    "Web Attack (XSS, SQL Injection, Brute Force)",
    "Infiltration",
    "Botnet",
    "Heartbleed",
]

PROJECT_OBJECTIVES = [
    "Detect and classify malicious network traffic using the CICIDS2017 "
    "benchmark dataset.",
    "Compare multiple machine learning models to identify the best "
    "performing intrusion classifier.",
    "Provide an interactive dashboard for real-time-style prediction and "
    "exploratory analytics.",
    "Deliver a reproducible, production-quality ML pipeline suitable for "
    "an enterprise security operations context.",
]


def _load_dataset_summary() -> Optional[Dict[str, Any]]:
    """
    Attempt to load a lightweight, pre-computed dataset summary.

    Looks for a JSON file (e.g. produced by the EDA/preprocessing stage)
    at ``<PROCESSED_DATA_DIR>/dataset_summary.json``. This keeps the
    dashboard fast by avoiding loading the full CICIDS2017 dataset into
    memory just to render a few headline numbers.

    Returns:
        A dictionary of summary statistics if the file exists and is
        valid JSON, otherwise None.
    """
    summary_path = Path(PROCESSED_DATA_DIR) / "dataset_summary.json"
    try:
        if not summary_path.exists():
            logger.info("Dataset summary file not found at %s", summary_path)
            return None
        with open(summary_path, "r", encoding="utf-8") as file_handle:
            summary = json.load(file_handle)
        logger.info("Loaded dataset summary from %s", summary_path)
        return summary
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load dataset summary: %s", exc)
        return None


def _render_key_metrics(summary: Optional[Dict[str, Any]]) -> None:
    """
    Render a row of headline metrics using Streamlit's metric widgets.

    If no pre-computed summary is available, informative placeholders
    are shown instead of raising an error, so the page remains usable
    before the preprocessing pipeline has been run.

    Args:
        summary: Optional dictionary containing keys such as
            'total_records', 'total_features', 'attack_ratio', and
            'num_classes'.
    """
    col1, col2, col3, col4 = st.columns(4)

    if summary is None:
        col1.metric("Total Records", "—")
        col2.metric("Total Features", "—")
        col3.metric("Attack Classes", "—")
        col4.metric("Attack Ratio", "—")
        st.info(
            "Dataset summary not found. Run the preprocessing pipeline "
            "to populate `dataset_summary.json` for live statistics.",
            icon="ℹ️",
        )
        return

    try:
        col1.metric("Total Records", f"{summary.get('total_records', 0):,}")
        col2.metric("Total Features", summary.get("total_features", "—"))
        col3.metric("Attack Classes", summary.get("num_classes", "—"))
        attack_ratio = summary.get("attack_ratio")
        col4.metric(
            "Attack Ratio",
            f"{attack_ratio:.2%}" if isinstance(attack_ratio, (int, float)) else "—",
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Malformed dataset summary values: %s", exc)
        st.warning("Dataset summary contains unexpected values.")


def _render_attack_categories() -> None:
    """Render the known CICIDS2017 attack categories as a reference table."""
    try:
        categories_df = pd.DataFrame(
            {
                "Category": ATTACK_CATEGORIES,
                "Type": [
                    "Normal" if cat == "BENIGN" else "Malicious"
                    for cat in ATTACK_CATEGORIES
                ],
            }
        )
        st.dataframe(categories_df, use_container_width=True, hide_index=True)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render attack category table.")
        st.error("Unable to display attack category reference table.")


def render() -> None:
    """
    Render the Home page.

    This is the public entry point invoked by ``src/dashboard/app.py``.
    All exceptions are caught locally so that a failure on this page
    does not propagate and crash the rest of the dashboard.
    """
    try:
        st.title("🛡️ Enterprise Network Intrusion Intelligence System")
        st.markdown(
            "An academic machine learning system for detecting and "
            "classifying network intrusions using the **CICIDS2017** "
            "benchmark dataset."
        )

        st.subheader("📌 Project Objectives")
        for objective in PROJECT_OBJECTIVES:
            st.markdown(f"- {objective}")

        st.subheader("📈 Dataset Snapshot")
        summary = _load_dataset_summary()
        _render_key_metrics(summary)

        st.subheader("🗂️ Known Attack Categories (CICIDS2017)")
        _render_attack_categories()

        st.subheader("🧭 Where to go next")
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            st.markdown(
                "**🔍 Prediction** — Submit network flow features and get "
                "a real-time classification of benign vs. malicious traffic."
            )
        with nav_col2:
            st.markdown(
                "**📊 Analytics** — Explore dataset distributions, model "
                "performance metrics, and visual insights."
            )

        logger.info("Home page rendered successfully.")

    except Exception:  # noqa: BLE001 - page-level safety net
        logger.exception("Unhandled exception while rendering the Home page.")
        st.error(
            "An unexpected error occurred while loading the Home page. "
            "Please check the application logs for details."
        )