"""
eda.py
=======

Exploratory Data Analysis (EDA) module for the Enterprise Network Intrusion
Intelligence System. This module provides a structured, reusable set of
routines for profiling the CICIDS2017 (cleaned & preprocessed variant)
network traffic dataset: class distribution analysis, descriptive
statistics, missing/infinite value auditing, correlation analysis, and
feature distribution visualization. All generated figures are persisted to
the reports/figures directory for inclusion in project documentation.

Dataset reference:
    https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

Author: Member A - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class EDAError(Exception):
    """Raised when an unrecoverable error occurs during exploratory data analysis."""


# --------------------------------------------------------------------------- #
# Configuration Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class EDAConfig:
    """
    Configuration container for the exploratory data analysis module.

    Attributes:
        figures_dir: Directory where generated figures are saved.
        target_column: Name of the column containing the traffic label.
        max_features_to_plot: Maximum number of numeric features to include
            in distribution/boxplot visualizations, to keep report figures
            readable for high-dimensional datasets.
        correlation_method: Correlation method passed to
            ``pandas.DataFrame.corr`` (e.g., 'pearson', 'spearman').
        figure_dpi: Resolution (dots per inch) used when saving figures.
        figure_format: File extension/format used when saving figures
            (e.g., 'png', 'svg').
    """

    figures_dir: Path = Path("reports/figures")
    target_column: str = "Label"
    max_features_to_plot: int = 12
    correlation_method: str = "pearson"
    figure_dpi: int = 150
    figure_format: str = "png"


# --------------------------------------------------------------------------- #
# Core EDA Class
# --------------------------------------------------------------------------- #
class ExploratoryDataAnalyzer:
    """
    Provides a suite of exploratory data analysis routines for network
    intrusion traffic data, producing both console/log summaries and
    persisted visualization artifacts.

    Example:
        >>> config = EDAConfig(target_column="Label")
        >>> eda = ExploratoryDataAnalyzer(config)
        >>> eda.run_full_report(df)
    """

    def __init__(self, config: EDAConfig) -> None:
        """
        Initialize the ExploratoryDataAnalyzer.

        Args:
            config: An EDAConfig instance describing analysis behavior and
                output locations.
        """
        self.config = config
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ExploratoryDataAnalyzer initialized with config: %s", self.config)

    # ------------------------------------------------------------------- #
    # Helper: figure saving
    # ------------------------------------------------------------------- #
    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """
        Save a matplotlib Figure to the configured figures directory.

        Args:
            fig: The Figure object to persist.
            filename: Base filename (without extension) for the saved
                figure.

        Returns:
            The full path to the saved figure file.

        Raises:
            EDAError: If the figure cannot be written to disk.
        """
        output_path = self.config.figures_dir / f"{filename}.{self.config.figure_format}"
        try:
            fig.tight_layout()
            fig.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved figure: %s", output_path)
            return output_path
        except (OSError, ValueError) as exc:
            plt.close(fig)
            logger.error("Failed to save figure '%s': %s", filename, exc)
            raise EDAError(f"Failed to save figure '{filename}': {exc}") from exc

    # ------------------------------------------------------------------- #
    # Dataset overview
    # ------------------------------------------------------------------- #
    def dataset_overview(self, df: pd.DataFrame) -> dict:
        """
        Produce a high-level structural overview of the dataset: shape,
        dtypes, memory usage, and missing value counts.

        Args:
            df: DataFrame to summarize.

        Returns:
            Dictionary containing summary statistics.

        Raises:
            EDAError: If the DataFrame is empty.
        """
        if df.empty:
            raise EDAError("Cannot generate overview: DataFrame is empty.")

        logger.info("Generating dataset overview.")

        overview = {
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
            "n_missing_values": int(df.isna().sum().sum()),
            "n_duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": df.select_dtypes(include=[np.number]).shape[1],
            "non_numeric_columns": df.select_dtypes(exclude=[np.number]).shape[1],
        }

        logger.info("Dataset overview: %s", overview)
        return overview

    # ------------------------------------------------------------------- #
    # Class distribution
    # ------------------------------------------------------------------- #
    def plot_class_distribution(self, df: pd.DataFrame) -> Path:
        """
        Visualize the distribution of the target label column as a
        horizontal bar chart, highlighting class imbalance which is
        characteristic of CICIDS2017 (benign traffic vastly outnumbers
        individual attack classes).

        Args:
            df: DataFrame containing the target column.

        Returns:
            Path to the saved figure.

        Raises:
            EDAError: If the target column is not present in the
                DataFrame.
        """
        target = self.config.target_column
        if target not in df.columns:
            raise EDAError(f"Target column '{target}' not found in DataFrame.")

        logger.info("Plotting class distribution for column '%s'.", target)

        counts = df[target].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(counts))))
        counts.plot(kind="barh", ax=ax, color=sns.color_palette("deep", len(counts)))
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Class")
        ax.set_title("Traffic Class Distribution (CICIDS2017)")
        for i, value in enumerate(counts.values):
            ax.text(value, i, f" {value:,}", va="center", fontsize=8)

        return self._save_figure(fig, "class_distribution")

    # ------------------------------------------------------------------- #
    # Missing / infinite value audit
    # ------------------------------------------------------------------- #
    def plot_missing_values(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Visualize the count of missing values per column, limited to
        columns that actually contain missing data. Returns None if no
        missing values are present.

        Args:
            df: DataFrame to audit.

        Returns:
            Path to the saved figure, or None if no missing values exist.
        """
        logger.info("Auditing missing values.")
        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if missing_counts.empty:
            logger.info("No missing values detected; skipping missing-value plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(missing_counts))))
        missing_counts.plot(kind="barh", ax=ax, color="indianred")
        ax.set_xlabel("Missing Value Count")
        ax.set_title("Missing Values per Feature")
        ax.invert_yaxis()

        return self._save_figure(fig, "missing_values")

    # ------------------------------------------------------------------- #
    # Correlation heatmap
    # ------------------------------------------------------------------- #
    def plot_correlation_heatmap(
        self, df: pd.DataFrame, max_features: int = 40
    ) -> Path:
        """
        Generate a correlation heatmap for numeric features. If the number
        of numeric features exceeds ``max_features``, the features with
        the highest variance are selected to keep the heatmap legible.

        Args:
            df: DataFrame containing numeric features.
            max_features: Maximum number of features to include in the
                heatmap.

        Returns:
            Path to the saved figure.

        Raises:
            EDAError: If no numeric columns are available.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise EDAError("No numeric columns available for correlation heatmap.")

        if numeric_df.shape[1] > max_features:
            top_variance_cols = (
                numeric_df.var().sort_values(ascending=False).head(max_features).index
            )
            numeric_df = numeric_df[top_variance_cols]
            logger.info(
                "Limiting correlation heatmap to top %d highest-variance features.",
                max_features,
            )

        logger.info("Computing %s correlation matrix.", self.config.correlation_method)
        corr = numeric_df.corr(method=self.config.correlation_method)

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.2,
            cbar_kws={"shrink": 0.7},
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap")
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)

        return self._save_figure(fig, "correlation_heatmap")

    # ------------------------------------------------------------------- #
    # Feature distributions
    # ------------------------------------------------------------------- #
    def plot_feature_distributions(self, df: pd.DataFrame) -> Path:
        """
        Generate a grid of histograms for a subset of numeric features
        (highest variance, up to ``max_features_to_plot``) to inspect
        skewness, scale, and potential outliers.

        Args:
            df: DataFrame containing numeric features.

        Returns:
            Path to the saved figure.

        Raises:
            EDAError: If no numeric columns are available.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise EDAError("No numeric columns available for distribution plots.")

        n_features = min(self.config.max_features_to_plot, numeric_df.shape[1])
        top_cols = (
            numeric_df.var().sort_values(ascending=False).head(n_features).index
        )

        logger.info("Plotting distributions for %d feature(s).", n_features)

        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, col in enumerate(top_cols):
            sns.histplot(numeric_df[col], bins=50, kde=True, ax=axes[idx], color="steelblue")
            axes[idx].set_title(col, fontsize=9)
            axes[idx].set_xlabel("")

        for idx in range(len(top_cols), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle("Feature Distributions (Top Variance Features)", y=1.02)

        return self._save_figure(fig, "feature_distributions")

    # ------------------------------------------------------------------- #
    # Boxplots by class
    # ------------------------------------------------------------------- #
    def plot_feature_boxplots_by_class(
        self, df: pd.DataFrame, top_n_features: int = 6
    ) -> Path:
        """
        Generate boxplots of the top-variance numeric features grouped by
        target class, useful for visually inspecting class-separability.

        Args:
            df: DataFrame containing numeric features and the target
                column.
            top_n_features: Number of top-variance features to visualize.

        Returns:
            Path to the saved figure.

        Raises:
            EDAError: If the target column is missing or no numeric
                columns are available.
        """
        target = self.config.target_column
        if target not in df.columns:
            raise EDAError(f"Target column '{target}' not found in DataFrame.")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise EDAError("No numeric columns available for boxplots.")

        top_cols = (
            numeric_df.var().sort_values(ascending=False).head(top_n_features).index
        )

        logger.info(
            "Plotting class-wise boxplots for %d feature(s).", len(top_cols)
        )

        n_cols = 2
        n_rows = int(np.ceil(len(top_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        plot_df = df[[target, *top_cols]].copy()

        for idx, col in enumerate(top_cols):
            sns.boxplot(
                data=plot_df, x=target, y=col, ax=axes[idx], showfliers=False
            )
            axes[idx].set_title(f"{col} by Class", fontsize=9)
            axes[idx].tick_params(axis="x", rotation=90, labelsize=6)

        for idx in range(len(top_cols), len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle("Feature Distributions by Traffic Class", y=1.02)

        return self._save_figure(fig, "feature_boxplots_by_class")

    # ------------------------------------------------------------------- #
    # Summary statistics export
    # ------------------------------------------------------------------- #
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptive statistics (count, mean, std, min, quartiles,
        max) for all numeric features.

        Args:
            df: DataFrame to summarize.

        Returns:
            DataFrame of descriptive statistics, transposed for
            readability (one row per feature).

        Raises:
            EDAError: If no numeric columns are available.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise EDAError("No numeric columns available for summary statistics.")

        logger.info("Generating descriptive summary statistics.")
        return numeric_df.describe().transpose()

    # ------------------------------------------------------------------- #
    # Orchestrator
    # ------------------------------------------------------------------- #
    def run_full_report(self, df: pd.DataFrame) -> dict:
        """
        Execute the full EDA suite: overview, class distribution, missing
        value audit, correlation heatmap, feature distributions, and
        class-wise boxplots. All figures are persisted to disk.

        Args:
            df: DataFrame to analyze (expected to be cleaned but not yet
                scaled/encoded, so the target column remains human
                readable).

        Returns:
            Dictionary containing the dataset overview and paths to all
            generated figures.

        Raises:
            EDAError: Propagated from any analysis stage on unrecoverable
                failure.
        """
        logger.info("=== Starting exploratory data analysis report ===")

        report: dict = {"overview": self.dataset_overview(df)}
        figure_paths: dict = {}

        try:
            figure_paths["class_distribution"] = str(self.plot_class_distribution(df))
        except EDAError as exc:
            logger.warning("Skipping class distribution plot: %s", exc)

        missing_fig = self.plot_missing_values(df)
        if missing_fig is not None:
            figure_paths["missing_values"] = str(missing_fig)

        try:
            figure_paths["correlation_heatmap"] = str(self.plot_correlation_heatmap(df))
        except EDAError as exc:
            logger.warning("Skipping correlation heatmap: %s", exc)

        try:
            figure_paths["feature_distributions"] = str(
                self.plot_feature_distributions(df)
            )
        except EDAError as exc:
            logger.warning("Skipping feature distributions plot: %s", exc)

        try:
            figure_paths["feature_boxplots_by_class"] = str(
                self.plot_feature_boxplots_by_class(df)
            )
        except EDAError as exc:
            logger.warning("Skipping class-wise boxplots: %s", exc)

        report["figures"] = figure_paths
        logger.info("=== Exploratory data analysis report completed ===")
        return report


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Command-line entry point for running the EDA report standalone, e.g.:

        python -m src.visualization.eda --input data/raw/cicids2017.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on the CICIDS2017 dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the CSV file to analyze.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Label",
        help="Name of the target/label column.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="reports/figures",
        help="Directory where generated figures will be saved.",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]

        config = EDAConfig(
            figures_dir=Path(args.figures_dir), target_column=args.target_column
        )
        analyzer = ExploratoryDataAnalyzer(config)
        report = analyzer.run_full_report(df)
        logger.info("EDA report summary: %s", report)

    except (EDAError, OSError, pd.errors.ParserError) as exc:
        logger.critical("EDA report generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()