"""
feature_engineering.py
=======================

Enterprise Network Intrusion Intelligence System
--------------------------------------------------
Feature engineering module responsible for selecting the most
predictive features from the preprocessed CICIDS2017 dataset using
Random Forest feature importances, and producing the engineered
feature matrices consumed by the training and evaluation stages.

Pipeline position
------------------
    data_preprocessing.py
        data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
            |
            v
    feature_engineering.py   <-- this module
        data/processed/X_train_fe.csv, X_test_fe.csv, feature_importances.csv
            |
            v
    train_model.py -> evaluate_model.py

Method
------
Feature selection uses ONLY Random Forest ``feature_importances_``:
    1. Fit a Random Forest classifier on the full training set
       (features vs. the "Attack Type" target).
    2. Rank all features by importance, descending.
    3. Keep the top 30 most important features.
    4. Apply the exact same selected feature set (same names, same
       order) to both the training and test feature matrices.

No PCA, RFE, SelectKBest, Sequential Feature Selection, or mutual
information selection is used, per project requirements.

Memory considerations
----------------------
The training set contains over two million rows. This module:
    - Loads each CSV exactly once.
    - Avoids concatenating the full training and test feature sets.
    - Avoids retaining unnecessary intermediate DataFrame copies
      (large temporary objects are deleted and garbage-collected
      as soon as they are no longer needed).
    - Uses column-view based access wherever possible before the
      unavoidable copy required to write the final engineered CSVs.

This module does NOT modify ``y_train.csv`` or ``y_test.csv``.

Author: Member B
Python Version: 3.11
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
try:
    from src.utils.logger import get_logger  # type: ignore

    logger = get_logger(__name__)
except ImportError:  # pragma: no cover - fallback path
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        _handler = logging.StreamHandler(sys.stdout)
        _formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class FeatureEngineeringError(Exception):
    """Base exception for all errors raised by this module."""


class MissingArtifactError(FeatureEngineeringError):
    """Raised when a required upstream preprocessing file is absent."""


class DataValidationError(FeatureEngineeringError):
    """Raised when input data fails validation checks."""


class SelectionError(FeatureEngineeringError):
    """Raised when feature importance ranking/selection fails."""


class PersistenceError(FeatureEngineeringError):
    """Raised when engineered outputs cannot be saved to disk."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureEngineeringConfig:
    """
    Configuration container for the :class:`FeatureEngineer`.

    Attributes
    ----------
    processed_data_dir : Path
        Directory containing both the upstream preprocessing outputs
        and the destination for engineered outputs.
    x_train_file : str
        Filename of the raw (preprocessed) training feature matrix.
    x_test_file : str
        Filename of the raw (preprocessed) test feature matrix.
    y_train_file : str
        Filename of the training labels.
    y_test_file : str
        Filename of the test labels.
    x_train_fe_file : str
        Output filename for the engineered training feature matrix.
    x_test_fe_file : str
        Output filename for the engineered test feature matrix.
    importances_file : str
        Output filename for the feature importance report.
    target_column : str
        Name of the target/label column within the y_* files.
    top_n_features : int
        Number of top-ranked features to retain.
    rf_n_estimators : int
        Number of trees used by the Random Forest fitted solely for
        importance ranking (not persisted, not the final model).
    rf_max_depth : Optional[int]
        Maximum tree depth for the importance-ranking Random Forest.
        Bounded to keep memory/time reasonable on 2M+ rows.
    random_state : int
        Seed used for reproducibility.
    n_jobs : int
        Number of parallel jobs for the Random Forest fit.
        ``-1`` uses all available cores.
    """

    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    x_train_file: str = "X_train.csv"
    x_test_file: str = "X_test.csv"
    y_train_file: str = "y_train.csv"
    y_test_file: str = "y_test.csv"
    x_train_fe_file: str = "X_train_fe.csv"
    x_test_fe_file: str = "X_test_fe.csv"
    importances_file: str = "feature_importances.csv"
    target_column: str = "Attack Type"
    top_n_features: int = 30
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 20
    random_state: int = 42
    n_jobs: int = -1

    @property
    def x_train_path(self) -> Path:
        """Full path to the raw training feature matrix."""
        return self.processed_data_dir / self.x_train_file

    @property
    def x_test_path(self) -> Path:
        """Full path to the raw test feature matrix."""
        return self.processed_data_dir / self.x_test_file

    @property
    def y_train_path(self) -> Path:
        """Full path to the training labels."""
        return self.processed_data_dir / self.y_train_file

    @property
    def y_test_path(self) -> Path:
        """Full path to the test labels."""
        return self.processed_data_dir / self.y_test_file

    @property
    def x_train_fe_path(self) -> Path:
        """Full output path for the engineered training feature matrix."""
        return self.processed_data_dir / self.x_train_fe_file

    @property
    def x_test_fe_path(self) -> Path:
        """Full output path for the engineered test feature matrix."""
        return self.processed_data_dir / self.x_test_fe_file

    @property
    def importances_path(self) -> Path:
        """Full output path for the feature importance report."""
        return self.processed_data_dir / self.importances_file

    def required_input_files(self) -> List[Path]:
        """
        Return the list of upstream files required before feature
        engineering can run.

        Returns
        -------
        List[Path]
            All required preprocessing output paths.
        """
        return [self.x_train_path, self.x_test_path, self.y_train_path, self.y_test_path]


# --------------------------------------------------------------------------- #
# FeatureEngineer
# --------------------------------------------------------------------------- #
class FeatureEngineer:
    """
    Selects the top-N most predictive features using Random Forest
    feature importances and produces engineered feature matrices for
    training and evaluation.

    The Random Forest fitted in this class is used exclusively to
    rank feature importance; it is intentionally lightweight and is
    NOT persisted, and it is NOT the final classification model
    trained by ``train_model.py``.

    Parameters
    ----------
    config : Optional[FeatureEngineeringConfig]
        Configuration controlling file locations and selection
        behavior. Defaults to standard project paths.

    Examples
    --------
    >>> engineer = FeatureEngineer()
    >>> engineer.run()
    """

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        self.config: FeatureEngineeringConfig = config or FeatureEngineeringConfig()
        self.selected_features: List[str] = []
        self.importances_df: Optional[pd.DataFrame] = None

        logger.info(
            "FeatureEngineer initialized | top_n_features=%d | "
            "processed_data_dir='%s'",
            self.config.top_n_features,
            self.config.processed_data_dir.resolve(),
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate_inputs(self) -> None:
        """
        Verify that every upstream preprocessing output required for
        feature engineering exists on disk.

        Raises
        ------
        MissingArtifactError
            If one or more required files are missing.
        """
        missing = [
            str(p) for p in self.config.required_input_files() if not p.exists()
        ]

        if missing:
            formatted = "\n  - ".join(missing)
            raise MissingArtifactError(
                "Feature engineering cannot proceed: the following required "
                f"files are missing:\n  - {formatted}\n\n"
                "Ensure data_preprocessing.py has been run first."
            )

        logger.info("All required preprocessing inputs were found.")

    # ------------------------------------------------------------------ #
    # Feature importance ranking
    # ------------------------------------------------------------------ #
    def _rank_feature_importances(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """
        Fit a Random Forest on the training data and rank all features
        by importance, descending.

        Parameters
        ----------
        x_train : pd.DataFrame
            Raw (preprocessed) training feature matrix.
        y_train : pd.Series
            Training target labels (string class names).

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame with ``feature`` and ``importance``,
            sorted by importance descending.

        Raises
        ------
        SelectionError
            If the Random Forest fails to fit or importances cannot
            be extracted.
        """
        try:
            non_numeric = x_train.select_dtypes(exclude=["number"]).columns.tolist()
            if non_numeric:
                raise DataValidationError(
                    f"Non-numeric feature columns detected: {non_numeric}. "
                    "Ensure data_preprocessing.py has fully encoded all "
                    "feature columns."
                )

            # Local, non-persisted label encoding solely to allow the
            # Random Forest importance fit; this encoder is discarded
            # after use and is NOT the encoder saved by train_model.py.
            local_encoder = LabelEncoder()
            y_encoded = local_encoder.fit_transform(y_train)

            logger.info(
                "Fitting Random Forest for feature importance ranking | "
                "n_estimators=%d | max_depth=%s | rows=%d | columns=%d",
                self.config.rf_n_estimators,
                self.config.rf_max_depth,
                x_train.shape[0],
                x_train.shape[1],
            )

            start_time = time.time()
            importance_rf = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            importance_rf.fit(x_train, y_encoded)
            elapsed = time.time() - start_time

            importances_df = pd.DataFrame(
                {
                    "feature": x_train.columns,
                    "importance": importance_rf.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False, ignore_index=True)

            logger.info(
                "Feature importance ranking complete in %.2fs | "
                "top_feature='%s' (%.4f)",
                elapsed,
                importances_df.iloc[0]["feature"],
                importances_df.iloc[0]["importance"],
            )

            # Free the importance-only model and encoder explicitly; they
            # are not needed beyond this point and the RF can be large.
            del importance_rf, local_encoder, y_encoded
            gc.collect()

            return importances_df

        except DataValidationError:
            logger.exception("Feature importance ranking aborted: invalid input.")
            raise
        except Exception as exc:
            logger.exception("Failed to compute feature importances.")
            raise SelectionError(f"Feature importance ranking failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select_top_features(self, importances_df: pd.DataFrame) -> List[str]:
        """
        Select the top-N most important feature names.

        Parameters
        ----------
        importances_df : pd.DataFrame
            Output of :meth:`_rank_feature_importances`, sorted
            descending by importance.

        Returns
        -------
        List[str]
            Ordered list of the top-N selected feature names.

        Raises
        ------
        SelectionError
            If fewer features are available than requested.
        """
        available = len(importances_df)
        if available < self.config.top_n_features:
            raise SelectionError(
                f"Requested top_n_features={self.config.top_n_features}, but "
                f"only {available} features are available."
            )

        top_features = importances_df.head(self.config.top_n_features)[
            "feature"
        ].tolist()

        logger.info(
            "Selected top %d features (of %d total).",
            len(top_features),
            available,
        )
        return top_features

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_outputs(
        self,
        x_train_fe: pd.DataFrame,
        x_test_fe: pd.DataFrame,
        importances_df: pd.DataFrame,
    ) -> None:
        """
        Persist the engineered training/test feature matrices and the
        full feature importance report to disk.

        Parameters
        ----------
        x_train_fe : pd.DataFrame
            Engineered training feature matrix (top-N columns only).
        x_test_fe : pd.DataFrame
            Engineered test feature matrix (identical columns/order
            to ``x_train_fe``).
        importances_df : pd.DataFrame
            Full feature importance ranking (all features, not just
            the selected top-N), for transparency and reporting.

        Raises
        ------
        PersistenceError
            If any output file fails to save.
        """
        try:
            self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

            x_train_fe.to_csv(self.config.x_train_fe_path, index=False)
            logger.info(
                "Engineered training features saved to '%s' | shape=%s",
                self.config.x_train_fe_path,
                x_train_fe.shape,
            )

            x_test_fe.to_csv(self.config.x_test_fe_path, index=False)
            logger.info(
                "Engineered test features saved to '%s' | shape=%s",
                self.config.x_test_fe_path,
                x_test_fe.shape,
            )

            selected_set = set(x_train_fe.columns)
            report_df = importances_df.copy()
            report_df["selected"] = report_df["feature"].isin(selected_set)
            report_df.to_csv(self.config.importances_path, index=False)
            logger.info(
                "Feature importance report saved to '%s' | total_features=%d",
                self.config.importances_path,
                len(report_df),
            )

        except Exception as exc:
            logger.exception("Failed to save feature engineering outputs.")
            raise PersistenceError(f"Failed to save outputs: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> List[str]:
        """
        Execute the full feature engineering pipeline end to end:
        validate inputs, load data (each file exactly once), rank
        importances, select the top-N features, and persist outputs.

        Returns
        -------
        List[str]
            The ordered list of selected feature names.

        Raises
        ------
        FeatureEngineeringError
            If any stage of the pipeline fails.
        """
        pipeline_start = time.time()
        logger.info("Starting feature engineering pipeline.")

        try:
            self.validate_inputs()

            # Load each CSV exactly once.
            logger.info("Loading X_train from '%s'", self.config.x_train_path)
            x_train = pd.read_csv(self.config.x_train_path)

            logger.info("Loading y_train from '%s'", self.config.y_train_path)
            y_train_df = pd.read_csv(self.config.y_train_path)

            if self.config.target_column in y_train_df.columns:
                y_train = y_train_df[self.config.target_column]
            else:
                # Fall back to the first (and typically only) column if the
                # expected target column name is not present verbatim.
                y_train = y_train_df.iloc[:, 0]

            if len(x_train) != len(y_train):
                raise DataValidationError(
                    f"Row count mismatch between X_train ({len(x_train)}) and "
                    f"y_train ({len(y_train)})."
                )

            importances_df = self._rank_feature_importances(x_train, y_train)
            self.selected_features = self.select_top_features(importances_df)
            self.importances_df = importances_df

            # Build the engineered training matrix, then release the raw
            # training matrix and labels before loading the test set, to
            # avoid holding two multi-million-row feature sets at once.
            x_train_fe = x_train[self.selected_features].copy()
            del x_train, y_train, y_train_df
            gc.collect()

            logger.info("Loading X_test from '%s'", self.config.x_test_path)
            x_test = pd.read_csv(self.config.x_test_path)

            missing_in_test = set(self.selected_features) - set(x_test.columns)
            if missing_in_test:
                raise DataValidationError(
                    f"Selected features missing from X_test: {sorted(missing_in_test)}"
                )

            x_test_fe = x_test[self.selected_features].copy()
            del x_test
            gc.collect()

            self.save_outputs(x_train_fe, x_test_fe, importances_df)

            del x_train_fe, x_test_fe
            gc.collect()

            elapsed_total = time.time() - pipeline_start
            logger.info(
                "Feature engineering pipeline completed successfully in %.2fs.",
                elapsed_total,
            )
            return self.selected_features

        except FeatureEngineeringError:
            logger.exception("Feature engineering pipeline aborted.")
            raise
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Unexpected error in feature engineering pipeline.")
            raise FeatureEngineeringError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------------- #
def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for standalone execution.

    Parameters
    ----------
    argv : Optional[List[str]]
        Argument list to parse. Defaults to ``sys.argv[1:]`` when
        ``None``. Exposed for unit testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="feature_engineering.py",
        description=(
            "Select the top-N most important features via Random Forest "
            "feature_importances_ and produce X_train_fe.csv, "
            "X_test_fe.csv, and feature_importances.csv."
        ),
    )

    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessing outputs. Default: 'data/processed'.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Attack Type",
        help="Name of the target column in y_train.csv/y_test.csv. Default: 'Attack Type'.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=30,
        help="Number of top-ranked features to retain. Default: 30.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=100,
        help="Number of trees in the importance-ranking Random Forest. Default: 100.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=20,
        help="Max tree depth for the importance-ranking Random Forest. Default: 20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for the Random Forest fit. Default: -1 (all cores).",
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    """
    Execute the feature engineering pipeline as a standalone script.

    Parameters
    ----------
    argv : Optional[List[str]]
        Command-line arguments (excluding program name). Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    args = parse_arguments(argv)

    logger.info("=" * 70)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 70)

    try:
        config = FeatureEngineeringConfig(
            processed_data_dir=Path(args.processed_data_dir),
            target_column=args.target_column,
            top_n_features=args.top_n_features,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=args.rf_max_depth,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

        engineer = FeatureEngineer(config=config)
        selected = engineer.run()

        logger.info("Selected features (%d): %s", len(selected), selected)
        return 0

    except FeatureEngineeringError as exc:
        logger.error("Feature engineering aborted: %s", exc)
        return 1
    except Exception:  # pragma: no cover - defensive catch-all
        logger.exception("Feature engineering aborted due to an unexpected error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())