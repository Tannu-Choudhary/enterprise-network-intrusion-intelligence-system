"""
feature_engineering.py
========================

Feature engineering pipeline for the Enterprise Network Intrusion
Intelligence System. This module consumes the cleaned/encoded output of
``src.data.data_preprocessing`` and performs feature-space refinement:
low-variance filtering, correlation-based redundancy removal, model-based
feature importance ranking, optional dimensionality reduction, and
persistence of the final selected feature set for use by the training and
inference pipelines.

Dataset reference:
    https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

Author: Member A - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

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


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class FeatureEngineeringError(Exception):
    """Raised when an unrecoverable error occurs during feature engineering."""


class FeatureValidationError(FeatureEngineeringError):
    """Raised when input data fails validation prior to feature engineering."""


# --------------------------------------------------------------------------- #
# Configuration Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class FeatureEngineeringConfig:
    """
    Configuration container for the feature engineering pipeline.

    Attributes:
        processed_data_dir: Directory containing X_train.csv / X_test.csv /
            y_train.csv / y_test.csv produced by the preprocessing stage.
        models_dir: Directory where feature-engineering artifacts
            (selected_features.csv, PCA transformer) are persisted.
        variance_threshold: Minimum variance a feature must have to be
            retained. Features at or below this threshold are dropped as
            near-constant / non-informative.
        correlation_threshold: Absolute Pearson correlation above which one
            of a pair of features is considered redundant and dropped.
        top_k_features: Number of top features to retain based on
            Random Forest feature importance ranking. If None, all
            surviving features (after variance/correlation filtering) are
            kept.
        apply_pca: Whether to additionally fit a PCA transformer on the
            selected features for optional dimensionality reduction
            (persisted separately; does not replace the selected feature
            set by default).
        pca_variance_ratio: Target cumulative explained variance ratio used
            to determine the number of PCA components when apply_pca=True.
        random_state: Seed for reproducibility in the Random Forest
            importance estimator and PCA.
        n_estimators: Number of trees used by the Random Forest importance
            estimator.
    """

    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    top_k_features: Optional[int] = 30
    apply_pca: bool = False
    pca_variance_ratio: float = 0.95
    random_state: int = 42
    n_estimators: int = 100
    _reserved: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core Feature Engineering Class
# --------------------------------------------------------------------------- #
class FeatureEngineer:
    """
    Encapsulates feature selection and transformation logic for the
    intrusion detection feature space.

    Pipeline stages:
        1. Load processed train/test splits from disk.
        2. Validate structural integrity.
        3. Remove low/zero-variance features.
        4. Remove highly correlated redundant features.
        5. Rank remaining features by Random Forest importance and select
           the top-k.
        6. Optionally fit a PCA transformer for dimensionality reduction.
        7. Persist the final selected feature list and any fitted
           transformers.

    Example:
        >>> config = FeatureEngineeringConfig()
        >>> engineer = FeatureEngineer(config)
        >>> X_train_fe, X_test_fe = engineer.run()
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """
        Initialize the FeatureEngineer.

        Args:
            config: A FeatureEngineeringConfig instance describing
                pipeline behavior and file locations.
        """
        self.config = config
        self.selected_features_: list[str] = []
        self.feature_importances_: Optional[pd.Series] = None
        self.pca_: Optional[PCA] = None

        logger.debug("FeatureEngineer initialized with config: %s", self.config)

    # ------------------------------------------------------------------- #
    # Step 1: Load
    # ------------------------------------------------------------------- #
    def load_processed_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load the processed train/test splits produced by the preprocessing
        stage.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            FeatureEngineeringError: If any required file is missing or
                cannot be parsed.
        """
        directory = self.config.processed_data_dir
        logger.info("Loading processed data from: %s", directory)

        required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        missing = [f for f in required_files if not (directory / f).exists()]
        if missing:
            raise FeatureEngineeringError(
                f"Missing required processed data file(s): {missing} in {directory}"
            )

        try:
            X_train = pd.read_csv(directory / "X_train.csv")
            X_test = pd.read_csv(directory / "X_test.csv")
            y_train = pd.read_csv(directory / "y_train.csv").squeeze("columns")
            y_test = pd.read_csv(directory / "y_test.csv").squeeze("columns")

            logger.info(
                "Processed data loaded. X_train: %s, X_test: %s",
                X_train.shape,
                X_test.shape,
            )
            return X_train, X_test, y_train, y_test

        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("Failed to parse processed data files: %s", exc)
            raise FeatureEngineeringError(
                f"Failed to parse processed data files: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Step 2: Validate
    # ------------------------------------------------------------------- #
    @staticmethod
    def validate_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Validate that feature and target data are non-empty, aligned, and
        numeric.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.

        Raises:
            FeatureValidationError: If validation fails.
        """
        logger.info("Validating feature engineering inputs.")

        if X_train.empty:
            raise FeatureValidationError("X_train is empty.")

        if len(X_train) != len(y_train):
            raise FeatureValidationError(
                f"Row count mismatch: X_train has {len(X_train)} rows, "
                f"y_train has {len(y_train)} rows."
            )

        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise FeatureValidationError(
                f"X_train contains non-numeric columns: {non_numeric}"
            )

        logger.info("Input validation passed.")

    # ------------------------------------------------------------------- #
    # Step 3: Variance filtering
    # ------------------------------------------------------------------- #
    def remove_low_variance_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove features whose variance is at or below the configured
        threshold, as they carry little to no discriminative signal.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of (X_train, X_test) with low-variance columns removed.

        Raises:
            FeatureEngineeringError: If variance filtering removes all
                features.
        """
        logger.info(
            "Applying variance threshold filtering (threshold=%.4f).",
            self.config.variance_threshold,
        )
        try:
            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            selector.fit(X_train)

            retained_mask = selector.get_support()
            retained_cols = X_train.columns[retained_mask].tolist()
            dropped_cols = X_train.columns[~retained_mask].tolist()

            if dropped_cols:
                logger.info(
                    "Dropped %d low-variance feature(s): %s",
                    len(dropped_cols),
                    dropped_cols,
                )

            if not retained_cols:
                raise FeatureEngineeringError(
                    "All features were removed by variance thresholding; "
                    "consider lowering variance_threshold."
                )

            return X_train[retained_cols], X_test[retained_cols]

        except ValueError as exc:
            logger.error("Variance filtering failed: %s", exc)
            raise FeatureEngineeringError(f"Variance filtering failed: {exc}") from exc

    # ------------------------------------------------------------------- #
    # Step 4: Correlation filtering
    # ------------------------------------------------------------------- #
    def remove_correlated_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify pairs of features with absolute Pearson correlation above
        the configured threshold and drop one feature from each redundant
        pair, retaining the first-encountered column.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of (X_train, X_test) with redundant columns removed.
        """
        logger.info(
            "Removing highly correlated features (threshold=%.2f).",
            self.config.correlation_threshold,
        )
        try:
            corr_matrix = X_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]

            if to_drop:
                logger.info(
                    "Dropped %d redundant correlated feature(s): %s",
                    len(to_drop),
                    to_drop,
                )
                X_train = X_train.drop(columns=to_drop)
                X_test = X_test.drop(columns=to_drop)
            else:
                logger.info("No redundant correlated features found.")

            return X_train, X_test

        except Exception as exc:  # noqa: BLE001 - guard against unexpected corr issues
            logger.error("Correlation filtering failed: %s", exc)
            raise FeatureEngineeringError(
                f"Correlation filtering failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Step 5: Importance-based selection
    # ------------------------------------------------------------------- #
    def select_top_features_by_importance(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rank remaining features using a Random Forest classifier's
        impurity-based feature importances and retain the top-k features.

        Args:
            X_train: Training feature matrix (post variance/correlation
                filtering).
            X_test: Test feature matrix (post variance/correlation
                filtering).
            y_train: Encoded training target vector.

        Returns:
            Tuple of (X_train, X_test) restricted to the top-k most
            important features. If ``top_k_features`` is None or exceeds
            the number of available features, all features are retained.

        Raises:
            FeatureEngineeringError: If the importance model fails to fit.
        """
        logger.info("Ranking features by Random Forest importance.")
        try:
            rf = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            importances = pd.Series(
                rf.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)
            self.feature_importances_ = importances

            logger.info(
                "Top 10 features by importance:\n%s", importances.head(10).to_string()
            )

            k = self.config.top_k_features
            if k is None or k >= len(importances):
                selected = importances.index.tolist()
                logger.info(
                    "Retaining all %d available features (top_k_features not "
                    "restrictive).",
                    len(selected),
                )
            else:
                selected = importances.head(k).index.tolist()
                logger.info("Selected top %d feature(s) by importance.", k)

            self.selected_features_ = selected
            return X_train[selected], X_test[selected]

        except ValueError as exc:
            logger.error("Feature importance ranking failed: %s", exc)
            raise FeatureEngineeringError(
                f"Feature importance ranking failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Step 6: Optional PCA
    # ------------------------------------------------------------------- #
    def apply_pca_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optionally fit a PCA transformer on the selected feature set to
        further reduce dimensionality while preserving a target cumulative
        explained variance ratio.

        Args:
            X_train: Selected training feature matrix.
            X_test: Selected test feature matrix.

        Returns:
            Tuple of (X_train_pca, X_test_pca) as DataFrames with PCA
            component columns. Returns the input unchanged if
            ``apply_pca`` is False.

        Raises:
            FeatureEngineeringError: If PCA fitting fails.
        """
        if not self.config.apply_pca:
            logger.info("PCA disabled via config; skipping dimensionality reduction.")
            return X_train, X_test

        logger.info(
            "Applying PCA (target explained variance ratio=%.2f).",
            self.config.pca_variance_ratio,
        )
        try:
            self.pca_ = PCA(
                n_components=self.config.pca_variance_ratio,
                random_state=self.config.random_state,
                svd_solver="full",
            )
            X_train_pca = self.pca_.fit_transform(X_train)
            X_test_pca = self.pca_.transform(X_test)

            component_cols = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
            logger.info(
                "PCA reduced feature space from %d to %d components "
                "(explained variance: %.4f).",
                X_train.shape[1],
                X_train_pca.shape[1],
                float(np.sum(self.pca_.explained_variance_ratio_)),
            )

            return (
                pd.DataFrame(X_train_pca, columns=component_cols, index=X_train.index),
                pd.DataFrame(X_test_pca, columns=component_cols, index=X_test.index),
            )

        except ValueError as exc:
            logger.error("PCA transformation failed: %s", exc)
            raise FeatureEngineeringError(f"PCA transformation failed: {exc}") from exc

    # ------------------------------------------------------------------- #
    # Step 7: Persist
    # ------------------------------------------------------------------- #
    def save_artifacts(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """
        Persist the final selected feature list, feature importance
        scores, engineered datasets, and (if fitted) the PCA transformer
        to disk.

        Args:
            X_train: Final engineered training feature matrix.
            X_test: Final engineered test feature matrix.

        Raises:
            FeatureEngineeringError: If saving any artifact fails.
        """
        try:
            self.config.models_dir.mkdir(parents=True, exist_ok=True)
            self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

            pd.Series(self.selected_features_, name="feature").to_csv(
                self.config.models_dir / "selected_features.csv", index=False
            )
            logger.info("Selected feature list saved to selected_features.csv.")

            if self.feature_importances_ is not None:
                self.feature_importances_.rename("importance").to_csv(
                    self.config.processed_data_dir / "feature_importances.csv"
                )
                logger.info("Feature importances saved.")

            X_train.to_csv(
                self.config.processed_data_dir / "X_train_fe.csv", index=False
            )
            X_test.to_csv(
                self.config.processed_data_dir / "X_test_fe.csv", index=False
            )
            logger.info("Engineered feature sets saved (X_train_fe.csv, X_test_fe.csv).")

            if self.pca_ is not None:
                joblib.dump(self.pca_, self.config.models_dir / "pca_transformer.pkl")
                logger.info("PCA transformer artifact saved.")

        except OSError as exc:
            logger.error("Failed to save feature engineering artifacts: %s", exc)
            raise FeatureEngineeringError(
                f"Failed to save feature engineering artifacts: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Orchestrator
    # ------------------------------------------------------------------- #
    def run(self, persist: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full feature engineering pipeline end-to-end.

        Args:
            persist: If True, saves engineered datasets and fitted
                artifacts to disk. Set to False for in-memory/testing use.

        Returns:
            Tuple of (X_train_engineered, X_test_engineered).

        Raises:
            FeatureEngineeringError: Propagated from any pipeline stage on
                unrecoverable failure.
        """
        logger.info("=== Starting feature engineering pipeline ===")

        X_train, X_test, y_train, y_test = self.load_processed_data()
        self.validate_inputs(X_train, y_train)

        X_train, X_test = self.remove_low_variance_features(X_train, X_test)
        X_train, X_test = self.remove_correlated_features(X_train, X_test)
        X_train, X_test = self.select_top_features_by_importance(
            X_train, X_test, y_train
        )
        X_train, X_test = self.apply_pca_transform(X_train, X_test)

        if persist:
            self.save_artifacts(X_train, X_test)

        logger.info("=== Feature engineering pipeline completed successfully ===")
        return X_train, X_test


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Command-line entry point for running the feature engineering pipeline
    standalone, e.g.:

        python -m src.features.feature_engineering --top-k 30
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Engineer and select features for CICIDS2017 intrusion data."
    )
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed train/test CSV files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top features to retain by importance ranking.",
    )
    parser.add_argument(
        "--apply-pca",
        action="store_true",
        help="If set, additionally fit a PCA transformer on selected features.",
    )
    args = parser.parse_args()

    try:
        config = FeatureEngineeringConfig(
            processed_data_dir=Path(args.processed_data_dir),
            top_k_features=args.top_k,
            apply_pca=args.apply_pca,
        )
        engineer = FeatureEngineer(config)
        engineer.run(persist=True)
    except FeatureEngineeringError as exc:
        logger.critical("Feature engineering pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()