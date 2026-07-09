"""
about.py
========

About page for the Enterprise Network Intrusion Intelligence System
dashboard.

This module renders static, informational content about the project:
academic context, team, dataset provenance, technology stack, and
references. It performs no data loading from the ML pipeline and is
therefore the simplest page in the dashboard, but still follows the
project's logging and exception-handling conventions for consistency
and to surface any unexpected rendering errors gracefully.

Author: Member C
Python: 3.11
"""

from __future__ import annotations

import logging
from typing import Dict, List

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

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Static content
# --------------------------------------------------------------------------
PROJECT_NAME = "Enterprise Network Intrusion Intelligence System"

PROJECT_DESCRIPTION = (
    "This project is an academic machine learning system that detects and "
    "classifies network intrusions using the CICIDS2017 benchmark dataset. "
    "It combines a supervised learning pipeline with an interactive "
    "Streamlit dashboard for prediction and analytics, simulating a "
    "simplified enterprise-grade security monitoring tool."
)

TEAM_RESPONSIBILITIES: List[Dict[str, str]] = [
    {
        "member": "Member A",
        "focus": "Data Engineering & Visualization",
        "modules": (
            "data_preprocessing.py, feature_engineering.py, eda.py, plots.py"
        ),
    },
    {
        "member": "Member B",
        "focus": "Modeling & Evaluation",
        "modules": (
            "model_trainer.py, predictor.py, train_model.py, evaluate_model.py"
        ),
    },
    {
        "member": "Member C",
        "focus": "Interactive Dashboard",
        "modules": "app.py, home.py, prediction.py, analytics.py, about.py",
    },
]

TECH_STACK: List[str] = [
    "Python 3.11",
    "pandas & NumPy — data manipulation",
    "scikit-learn — model training & evaluation",
    "joblib — model artifact persistence",
    "matplotlib & seaborn — static visualizations",
    "Plotly — interactive visualizations",
    "Streamlit — dashboard framework",
]

DATASET_INFO: Dict[str, str] = {
    "Name": "CICIDS2017 (Cleaned & Preprocessed)",
    "Source": (
        "Kaggle — ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
    ),
    "Original Publisher": (
        "Canadian Institute for Cybersecurity (CIC), University of New "
        "Brunswick"
    ),
    "Description": (
        "Labeled network traffic flow data covering benign traffic and "
        "multiple attack categories (DoS, DDoS, PortScan, Brute Force, "
        "Web Attacks, Infiltration, Botnet, Heartbleed), used as a "
        "standard benchmark for intrusion detection research."
    ),
}

REFERENCES: List[str] = [
    "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). "
    "Toward Generating a New Intrusion Detection Dataset and Intrusion "
    "Traffic Characterization. ICISSP.",
    "Kaggle dataset: cicids2017-cleaned-and-preprocessed "
    "(ericanacletoribeiro).",
    "scikit-learn documentation — https://scikit-learn.org",
    "Streamlit documentation — https://docs.streamlit.io",
]


def _render_team_section() -> None:
    """Render the team responsibilities table."""
    st.subheader("👥 Team & Responsibilities")
    try:
        for entry in TEAM_RESPONSIBILITIES:
            with st.container(border=True):
                st.markdown(f"**{entry['member']}** — {entry['focus']}")
                st.caption(entry["modules"])
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render team section.")
        st.error("Unable to display team information.")


def _render_dataset_section() -> None:
    """Render dataset provenance and description."""
    st.subheader("🗂️ Dataset")
    try:
        for key, value in DATASET_INFO.items():
            st.markdown(f"**{key}:** {value}")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render dataset section.")
        st.error("Unable to display dataset information.")


def _render_tech_stack_section() -> None:
    """Render the technology stack used across the project."""
    st.subheader("🛠️ Technology Stack")
    try:
        for item in TECH_STACK:
            st.markdown(f"- {item}")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render tech stack section.")
        st.error("Unable to display technology stack.")


def _render_references_section() -> None:
    """Render academic and technical references."""
    st.subheader("📚 References")
    try:
        for index, reference in enumerate(REFERENCES, start=1):
            st.markdown(f"{index}. {reference}")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to render references section.")
        st.error("Unable to display references.")


def render() -> None:
    """
    Render the About page.

    This is the public entry point invoked by ``src/dashboard/app.py``.
    All exceptions are caught locally so that a failure on this page
    does not propagate and crash the rest of the dashboard.
    """
    try:
        st.title(f"ℹ️ About — {PROJECT_NAME}")
        st.markdown(PROJECT_DESCRIPTION)
        st.markdown("---")

        _render_team_section()
        st.markdown("---")

        _render_dataset_section()
        st.markdown("---")

        _render_tech_stack_section()
        st.markdown("---")

        _render_references_section()

        st.markdown("---")
        st.caption(
            "This dashboard was built for academic purposes as part of a "
            "university machine learning / cybersecurity coursework project."
        )

        logger.info("About page rendered successfully.")

    except Exception:  # noqa: BLE001 - page-level safety net
        logger.exception("Unhandled exception while rendering the About page.")
        st.error(
            "An unexpected error occurred while loading the About page. "
            "Please check the application logs for details."
        )