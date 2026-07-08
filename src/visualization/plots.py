"""
plots.py
=========

Reusable plotting utility library for the Enterprise Network Intrusion
Intelligence System. While ``eda.py`` focuses on raw-data exploration,
this module provides general-purpose, model-agnostic visualization
functions intended for reuse across the project (model evaluation reports,
dashboard visualizations, and documentation figures): confusion matrices,
ROC/Precision-Recall curves, feature importance charts, learning curves,
and interactive Plotly variants for dashboard embedding.

Functions are stateless and side-effect-light by design (they return
Figure objects and optionally save to disk), so they can be safely called
from static reporting scripts (e.g., evaluate_model.py) as well as from an
interactive Streamlit dashboard.

Author: Member A - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")  # Non-interactive backend suitable for headless/report generation

# --------------------------------------------------------------------------- #
# Logging Configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_console_handler)

sns.set_theme(style="whitegrid", palette="deep")

DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_DPI = 150
DEFAULT_FORMAT = "png"


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class PlottingError(Exception):
    """Raised when an unrecoverable error occurs while generating a plot."""


# --------------------------------------------------------------------------- #
# Internal Helpers
# --------------------------------------------------------------------------- #
def _save_matplotlib_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    dpi: int = DEFAULT_DPI,
    fmt: str = DEFAULT_FORMAT,
) -> Path:
    """
    Persist a matplotlib Figure to disk under the given output directory.

    Args:
        fig: The Figure to save.
        filename: Base filename (without extension).
        output_dir: Directory to save the figure into; created if absent.
        dpi: Resolution in dots per inch.
        fmt: File format/extension (e.g., 'png', 'svg').

    Returns:
        Path to the saved figure file.

    Raises:
        PlottingError: If the figure cannot be written to disk.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.{fmt}"
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure: %s", output_path)
        return output_path
    except (OSError, ValueError) as exc:
        plt.close(fig)
        logger.error("Failed to save figure '%s': %s", filename, exc)
        raise PlottingError(f"Failed to save figure '{filename}': {exc}") from exc


def _validate_prediction_inputs(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_pred: Union[np.ndarray, pd.Series, Sequence],
) -> None:
    """
    Validate that ground-truth and prediction arrays are non-empty and of
    equal length.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Raises:
        PlottingError: If validation fails.
    """
    if y_true is None or y_pred is None:
        raise PlottingError("y_true and y_pred must not be None.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise PlottingError("y_true and y_pred must not be empty.")
    if len(y_true) != len(y_pred):
        raise PlottingError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)} elements."
        )


# --------------------------------------------------------------------------- #
# Confusion Matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_pred: Union[np.ndarray, pd.Series, Sequence],
    class_names: Optional[Sequence[str]] = None,
    normalize: Optional[str] = "true",
    output_dir: Path = DEFAULT_FIGURES_DIR,
    filename: str = "confusion_matrix",
    save: bool = True,
) -> plt.Figure:
    """
    Generate a confusion matrix heatmap comparing true vs. predicted
    intrusion class labels.

    Args:
        y_true: Ground-truth encoded labels.
        y_pred: Model-predicted encoded labels.
        class_names: Optional display names for each class, ordered by
            encoded label index.
        normalize: Normalization mode passed to
            ``sklearn.metrics.confusion_matrix`` ('true', 'pred', 'all',
            or None for raw counts).
        output_dir: Directory to save the figure into.
        filename: Base filename for the saved figure.
        save: Whether to persist the figure to disk.

    Returns:
        The matplotlib Figure object.

    Raises:
        PlottingError: If inputs are invalid or the matrix cannot be
            computed.
    """
    _validate_prediction_inputs(y_true, y_pred)
    logger.info("Generating confusion matrix (normalize=%s).", normalize)

    try:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        fig, ax = plt.subplots(figsize=(9, 8))
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        display.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")
        ax.set_title("Confusion Matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

        if save:
            _save_matplotlib_figure(fig, filename, output_dir)
        return fig

    except ValueError as exc:
        logger.error("Failed to generate confusion matrix: %s", exc)
        raise PlottingError(f"Failed to generate confusion matrix: {exc}") from exc


# --------------------------------------------------------------------------- #
# ROC Curve
# --------------------------------------------------------------------------- #
def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_score: Union[np.ndarray, pd.Series, Sequence],
    pos_label: Optional[Union[int, str]] = None,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    filename: str = "roc_curve",
    save: bool = True,
) -> plt.Figure:
    """
    Generate a Receiver Operating Characteristic (ROC) curve with the
    associated Area Under the Curve (AUC) for a binary classification
    task (e.g., benign vs. attack).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability or decision score for the positive
            class.
        pos_label: Label of the positive class. Required if labels are
            not already {0, 1}.
        output_dir: Directory to save the figure into.
        filename: Base filename for the saved figure.
        save: Whether to persist the figure to disk.

    Returns:
        The matplotlib Figure object.

    Raises:
        PlottingError: If inputs are invalid or the curve cannot be
            computed.
    """
    _validate_prediction_inputs(y_true, y_score)
    logger.info("Generating ROC curve.")

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 7))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        logger.info("ROC AUC: %.4f", roc_auc)

        if save:
            _save_matplotlib_figure(fig, filename, output_dir)
        return fig

    except ValueError as exc:
        logger.error("Failed to generate ROC curve: %s", exc)
        raise PlottingError(f"Failed to generate ROC curve: {exc}") from exc


# --------------------------------------------------------------------------- #
# Precision-Recall Curve
# --------------------------------------------------------------------------- #
def plot_precision_recall_curve(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_score: Union[np.ndarray, pd.Series, Sequence],
    pos_label: Optional[Union[int, str]] = None,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    filename: str = "precision_recall_curve",
    save: bool = True,
) -> plt.Figure:
    """
    Generate a Precision-Recall curve, particularly informative for
    imbalanced intrusion detection datasets where attack classes are
    heavily outnumbered by benign traffic.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probability or decision score for the positive
            class.
        pos_label: Label of the positive class. Required if labels are
            not already {0, 1}.
        output_dir: Directory to save the figure into.
        filename: Base filename for the saved figure.
        save: Whether to persist the figure to disk.

    Returns:
        The matplotlib Figure object.

    Raises:
        PlottingError: If inputs are invalid or the curve cannot be
            computed.
    """
    _validate_prediction_inputs(y_true, y_score)
    logger.info("Generating precision-recall curve.")

    try:
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label
        )
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower left")

        logger.info("Precision-Recall AUC: %.4f", pr_auc)

        if save:
            _save_matplotlib_figure(fig, filename, output_dir)
        return fig

    except ValueError as exc:
        logger.error("Failed to generate precision-recall curve: %s", exc)
        raise PlottingError(
            f"Failed to generate precision-recall curve: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Feature Importance
# --------------------------------------------------------------------------- #
def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 20,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    filename: str = "feature_importance",
    save: bool = True,
) -> plt.Figure:
    """
    Generate a horizontal bar chart of the top-N most important features
    according to a fitted model's importance scores.

    Args:
        feature_names: Sequence of feature names.
        importances: Sequence of importance scores aligned with
            ``feature_names``.
        top_n: Number of top features to display.
        output_dir: Directory to save the figure into.
        filename: Base filename for the saved figure.
        save: Whether to persist the figure to disk.

    Returns:
        The matplotlib Figure object.

    Raises:
        PlottingError: If input lengths mismatch or are empty.
    """
    if len(feature_names) != len(importances):
        raise PlottingError(
            f"Length mismatch: {len(feature_names)} feature names vs. "
            f"{len(importances)} importance scores."
        )
    if len(feature_names) == 0:
        raise PlottingError("feature_names and importances must not be empty.")
    if top_n <= 0:
        raise PlottingError(f"top_n must be greater than 0, got {top_n}.")
    if top_n > len(feature_names):
        logger.warning(
            "top_n (%d) exceeds the number of available features (%d); "
            "all features will be plotted.",
            top_n,
            len(feature_names),
        )

    logger.info("Generating feature importance plot (top_n=%d).", top_n)

    try:
        series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        series = series.head(min(top_n, len(series))).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(series))))
        series.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(series)))
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {len(series)} Feature Importances")

        if save:
            _save_matplotlib_figure(fig, filename, output_dir)
        return fig

    except (ValueError, KeyError) as exc:
        logger.error("Failed to generate feature importance plot: %s", exc)
        raise PlottingError(
            f"Failed to generate feature importance plot: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Learning Curve
# --------------------------------------------------------------------------- #
def plot_learning_curve(
    train_sizes: Sequence[float],
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    metric_name: str = "Score",
    output_dir: Path = DEFAULT_FIGURES_DIR,
    filename: str = "learning_curve",
    save: bool = True,
) -> plt.Figure:
    """
    Generate a learning curve plot showing training and validation
    performance as a function of training set size, useful for diagnosing
    over/underfitting.

    Args:
        train_sizes: Sequence of training set sizes evaluated.
        train_scores: 2D array of shape (n_sizes, n_cv_folds) with
            training scores per fold.
        val_scores: 2D array of shape (n_sizes, n_cv_folds) with
            validation scores per fold.
        metric_name: Display name of the evaluation metric (e.g.,
            'F1-Score', 'Accuracy').
        output_dir: Directory to save the figure into.
        filename: Base filename for the saved figure.
        save: Whether to persist the figure to disk.

    Returns:
        The matplotlib Figure object.

    Raises:
        PlottingError: If inputs are malformed.
    """
    try:
        train_scores = np.asarray(train_scores)
        val_scores = np.asarray(val_scores)

        if train_scores.ndim != 2 or val_scores.ndim != 2:
            raise PlottingError(
                "train_scores and val_scores must be 2D arrays of shape "
                "(n_sizes, n_cv_folds)."
            )

        if train_scores.size == 0 or val_scores.size == 0:
            raise PlottingError("train_scores and val_scores must not be empty.")

        if (
            len(train_sizes) != train_scores.shape[0]
            or len(train_sizes) != val_scores.shape[0]
        ):
            raise PlottingError(
                f"Length mismatch: train_sizes has {len(train_sizes)} entries, "
                f"but train_scores has {train_scores.shape[0]} row(s) and "
                f"val_scores has {val_scores.shape[0]} row(s)."
            )

        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        logger.info("Generating learning curve for metric '%s'.", metric_name)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training")
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color="steelblue",
        )
        ax.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation")
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color="darkorange",
        )
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Learning Curve ({metric_name})")
        ax.legend(loc="best")

        if save:
            _save_matplotlib_figure(fig, filename, output_dir)
        return fig

    except (ValueError, TypeError) as exc:
        logger.error("Failed to generate learning curve: %s", exc)
        raise PlottingError(f"Failed to generate learning curve: {exc}") from exc


# --------------------------------------------------------------------------- #
# Interactive Plotly Variants (for dashboard embedding)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix_interactive(
    y_true: Union[np.ndarray, pd.Series, Sequence],
    y_pred: Union[np.ndarray, pd.Series, Sequence],
    class_names: Optional[Sequence[str]] = None,
    normalize: Optional[str] = "true",
) -> go.Figure:
    """
    Generate an interactive Plotly confusion matrix heatmap suitable for
    embedding in the Streamlit dashboard.

    Args:
        y_true: Ground-truth encoded labels.
        y_pred: Model-predicted encoded labels.
        class_names: Optional display names for each class.
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        A Plotly Figure object.

    Raises:
        PlottingError: If inputs are invalid or the matrix cannot be
            computed.
    """
    _validate_prediction_inputs(y_true, y_pred)
    logger.info("Generating interactive confusion matrix.")

    try:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        labels = (
            list(class_names)
            if class_names is not None
            else [str(i) for i in range(cm.shape[0])]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=np.round(cm, 2),
                texttemplate="%{text}",
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis_autorange="reversed",
        )
        return fig

    except ValueError as exc:
        logger.error("Failed to generate interactive confusion matrix: %s", exc)
        raise PlottingError(
            f"Failed to generate interactive confusion matrix: {exc}"
        ) from exc


def plot_feature_importance_interactive(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 20,
) -> go.Figure:
    """
    Generate an interactive Plotly horizontal bar chart of the top-N
    feature importances, suitable for embedding in the Streamlit
    dashboard.

    Args:
        feature_names: Sequence of feature names.
        importances: Sequence of importance scores aligned with
            ``feature_names``.
        top_n: Number of top features to display.

    Returns:
        A Plotly Figure object.

    Raises:
        PlottingError: If input lengths mismatch or are empty.
    """
    if len(feature_names) != len(importances):
        raise PlottingError(
            f"Length mismatch: {len(feature_names)} feature names vs. "
            f"{len(importances)} importance scores."
        )
    if len(feature_names) == 0:
        raise PlottingError("feature_names and importances must not be empty.")
    if top_n <= 0:
        raise PlottingError(f"top_n must be greater than 0, got {top_n}.")
    if top_n > len(feature_names):
        logger.warning(
            "top_n (%d) exceeds the number of available features (%d); "
            "all features will be plotted.",
            top_n,
            len(feature_names),
        )

    logger.info("Generating interactive feature importance chart (top_n=%d).", top_n)

    try:
        series = pd.Series(importances, index=feature_names).sort_values(
            ascending=False
        ).head(min(top_n, len(feature_names)))
        series = series.sort_values(ascending=True)

        fig = go.Figure(
            go.Bar(
                x=series.values,
                y=series.index,
                orientation="h",
                marker=dict(color=series.values, colorscale="Viridis"),
            )
        )
        fig.update_layout(
            title=f"Top {len(series)} Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
        )
        return fig

    except (ValueError, KeyError) as exc:
        logger.error("Failed to generate interactive feature importance chart: %s", exc)
        raise PlottingError(
            f"Failed to generate interactive feature importance chart: {exc}"
        ) from exc


def plot_metric_comparison_interactive(
    model_names: Sequence[str],
    metric_values: dict[str, Sequence[float]],
) -> go.Figure:
    """
    Generate an interactive grouped bar chart comparing multiple
    evaluation metrics across several trained models, suitable for a
    dashboard model-comparison view.

    Args:
        model_names: Sequence of model names/labels (x-axis categories).
        metric_values: Mapping of metric name (e.g., 'Accuracy', 'F1')
            to a sequence of values aligned with ``model_names``.

    Returns:
        A Plotly Figure object.

    Raises:
        PlottingError: If any metric's value sequence length does not
            match ``model_names``.
    """
    if not model_names:
        raise PlottingError("model_names must not be empty.")
    if not metric_values:
        raise PlottingError("metric_values must not be empty.")

    for metric, values in metric_values.items():
        if len(values) != len(model_names):
            raise PlottingError(
                f"Metric '{metric}' has {len(values)} values but there are "
                f"{len(model_names)} model names."
            )

    logger.info(
        "Generating interactive metric comparison chart across %d model(s).",
        len(model_names),
    )

    try:
        fig = go.Figure()
        for metric, values in metric_values.items():
            fig.add_trace(go.Bar(name=metric, x=list(model_names), y=list(values)))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
        )
        return fig

    except (ValueError, KeyError) as exc:
        logger.error("Failed to generate metric comparison chart: %s", exc)
        raise PlottingError(
            f"Failed to generate metric comparison chart: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Self-Test Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Lightweight self-test that generates each plot type using synthetic
    data, useful for verifying the plotting library functions correctly
    without requiring a trained model or real dataset.
    """
    logger.info("Running plots.py self-test with synthetic data.")

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_score = np.clip(y_true + rng.normal(0, 0.4, size=500), 0, 1)
    y_pred = (y_score > 0.5).astype(int)

    try:
        plot_confusion_matrix(y_true, y_pred, class_names=["Benign", "Attack"])
        plot_roc_curve(y_true, y_score)
        plot_precision_recall_curve(y_true, y_score)
        plot_feature_importance(
            feature_names=[f"feature_{i}" for i in range(15)],
            importances=rng.random(15),
        )
        logger.info("Self-test completed successfully. Figures saved to %s", DEFAULT_FIGURES_DIR)
    except PlottingError as exc:
        logger.critical("Self-test failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()