"""
data_preprocessing.py
======================

Data preprocessing pipeline for the Enterprise Network Intrusion Intelligence
System. This module is responsible for loading the raw CICIDS2017
(cleaned & preprocessed variant) network traffic dataset, performing data
cleaning, handling missing/infinite values, encoding categorical labels,
scaling numerical features, splitting the data into train/test partitions,
and persisting both the processed datasets and the fitted transformation
artifacts (scaler, label encoder) for downstream use by the modeling and
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
class DataPreprocessingError(Exception):
    """Raised when an unrecoverable error occurs during data preprocessing."""


class DataValidationError(DataPreprocessingError):
    """Raised when the input dataset fails structural or content validation."""


# --------------------------------------------------------------------------- #
# Configuration Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingConfig:
    """
    Configuration container for the preprocessing pipeline.

    Attributes:
        raw_data_path: Path to the raw CICIDS2017 CSV file(s).
        processed_data_dir: Directory where processed artifacts are written.
        models_dir: Directory where fitted transformers (scaler, encoder)
            are persisted.
        target_column: Name of the column containing the traffic label
            (e.g., 'Label' or 'Attack_Type').
        test_size: Fraction of data reserved for the test split.
        random_state: Seed used for reproducibility across random operations.
        drop_duplicates: Whether duplicate rows should be removed.
        scale_features: Whether numeric features should be standardized.
        stratify: Whether the train/test split should be stratified on the
            target column.
        columns_to_drop: Optional list of known non-informative columns
            (e.g., 'Flow ID', 'Timestamp') to remove prior to modeling.
    """

    raw_data_path: Path
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    target_column: str = "Label"
    test_size: float = 0.2
    random_state: int = 42
    drop_duplicates: bool = True
    scale_features: bool = True
    stratify: bool = True
    columns_to_drop: list[str] = field(
        default_factory=lambda: [
            "Flow ID",
            "Source IP",
            "Src IP",
            "Destination IP",
            "Dst IP",
            "Timestamp",
            "SimillarHTTP",
            "Unnamed: 0",
        ]
    )


# --------------------------------------------------------------------------- #
# Core Preprocessing Class
# --------------------------------------------------------------------------- #
class DataPreprocessor:
    """
    Encapsulates the full data preprocessing workflow for CICIDS2017 network
    intrusion traffic data.

    The pipeline performs the following ordered steps:
        1. Load raw data from disk.
        2. Validate structural integrity (non-empty, target column present).
        3. Normalize column names (strip whitespace).
        4. Drop known non-informative / identifier columns.
        5. Handle infinite values by converting them to NaN.
        6. Handle missing values via row/column-level strategies.
        7. Remove duplicate records.
        8. Encode the target label column.
        9. Split features/target into train and test sets.
        10. Scale numeric features (fit on train, transform on test).
        11. Persist processed datasets and fitted artifacts to disk.

    Example:
        >>> config = PreprocessingConfig(raw_data_path=Path("data/raw/cicids2017.csv"))
        >>> preprocessor = DataPreprocessor(config)
        >>> X_train, X_test, y_train, y_test = preprocessor.run()
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        """
        Initialize the DataPreprocessor.

        Args:
            config: A PreprocessingConfig instance describing pipeline
                behavior and file locations.
        """
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._raw_df: Optional[pd.DataFrame] = None

        logger.debug("DataPreprocessor initialized with config: %s", self.config)

    # ------------------------------------------------------------------- #
    # Step 1: Load
    # ------------------------------------------------------------------- #
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset from the configured path.

        Supports a single CSV file or a directory of CSV files, which are
        concatenated into a single DataFrame (CICIDS2017 is often
        distributed as multiple per-day CSV files).

        Returns:
            The raw, unprocessed DataFrame.

        Raises:
            DataPreprocessingError: If the file/directory does not exist or
                no CSV files could be read.
        """
        path = self.config.raw_data_path
        logger.info("Loading raw data from: %s", path)

        try:
            if not path.exists():
                raise DataPreprocessingError(f"Raw data path does not exist: {path}")

            if path.is_dir():
                csv_files = sorted(path.glob("*.csv"))
                if not csv_files:
                    raise DataPreprocessingError(
                        f"No CSV files found in directory: {path}"
                    )
                logger.info("Found %d CSV file(s) to concatenate.", len(csv_files))
                frames = [pd.read_csv(f, low_memory=False) for f in csv_files]
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.read_csv(path, low_memory=False)

            logger.info(
                "Raw data loaded successfully. Shape: %s", df.shape
            )
            self._raw_df = df
            return df

        except pd.errors.EmptyDataError as exc:
            logger.error("The provided CSV file is empty.")
            raise DataPreprocessingError("The provided CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            logger.error("Failed to parse CSV file: %s", exc)
            raise DataPreprocessingError(f"Failed to parse CSV file: {exc}") from exc
        except OSError as exc:
            logger.error("OS error while reading raw data: %s", exc)
            raise DataPreprocessingError(
                f"OS error while reading raw data: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Step 2: Validate
    # ------------------------------------------------------------------- #
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the loaded DataFrame meets minimum structural
        requirements before processing continues.

        Args:
            df: The raw or partially processed DataFrame to validate.

        Raises:
            DataValidationError: If the DataFrame is empty or the target
                column cannot be located (case-insensitive, whitespace
                tolerant).
        """
        logger.info("Validating dataset structure.")

        if df.empty:
            raise DataValidationError("Loaded dataset is empty.")

        normalized_cols = {c.strip().lower(): c for c in df.columns}
        target_key = self.config.target_column.strip().lower()

        if target_key not in normalized_cols:
            raise DataValidationError(
                f"Target column '{self.config.target_column}' not found in "
                f"dataset columns: {list(df.columns)}"
            )

        # Resolve to the actual column name present in the DataFrame.
        actual_target_name = normalized_cols[target_key]
        if actual_target_name != self.config.target_column:
            logger.warning(
                "Target column resolved as '%s' (config specified '%s').",
                actual_target_name,
                self.config.target_column,
            )
            self.config.target_column = actual_target_name

        logger.info("Dataset validation passed. Shape: %s", df.shape)

    # ------------------------------------------------------------------- #
    # Step 3: Clean column names
    # ------------------------------------------------------------------- #
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace from column names, a common
        artifact in the CICIDS2017 CSV exports.

        Args:
            df: DataFrame whose columns require normalization.

        Returns:
            DataFrame with cleaned column names.
        """
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        logger.debug("Column names normalized.")
        return df

    # ------------------------------------------------------------------- #
    # Step 4: Drop non-informative columns
    # ------------------------------------------------------------------- #
    def drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove identifier and non-informative columns (e.g., Flow ID, IP
        addresses, timestamps) that provide no generalizable predictive
        signal and risk leaking identity information into the model.

        Args:
            df: DataFrame from which columns will be dropped.

        Returns:
            DataFrame with irrelevant columns removed.
        """
        cols_present = [c for c in self.config.columns_to_drop if c in df.columns]
        if cols_present:
            logger.info("Dropping irrelevant columns: %s", cols_present)
            df = df.drop(columns=cols_present)
        else:
            logger.info("No configured irrelevant columns found in dataset.")
        return df

    # ------------------------------------------------------------------- #
    # Step 5 & 6: Handle infinities and missing values
    # ------------------------------------------------------------------- #
    def handle_missing_and_infinite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinite values with NaN, then impute or drop missing
        values. Numeric columns are imputed with the column median;
        rows still containing missing values after imputation (e.g., in
        non-numeric columns) are dropped.

        Args:
            df: DataFrame to clean.

        Returns:
            DataFrame free of infinite and missing values.

        Raises:
            DataPreprocessingError: If cleaning results in an empty
                DataFrame.
        """
        logger.info("Handling infinite and missing values.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        initial_shape = df.shape

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        total_missing_before = int(df.isna().sum().sum())
        logger.info("Total missing/infinite cells detected: %d", total_missing_before)

        if total_missing_before > 0:
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.debug(
                        "Filled NaNs in column '%s' with median value %.4f",
                        col,
                        median_val,
                    )

        remaining_na = df.isna().sum().sum()
        if remaining_na > 0:
            logger.warning(
                "%d missing values remain in non-numeric columns; dropping "
                "affected rows.",
                remaining_na,
            )
            df = df.dropna()

        if df.empty:
            raise DataPreprocessingError(
                "Dataset became empty after handling missing/infinite values."
            )

        logger.info(
            "Missing/infinite value handling complete. Shape before: %s, after: %s",
            initial_shape,
            df.shape,
        )
        return df

    # ------------------------------------------------------------------- #
    # Step 7: Deduplicate
    # ------------------------------------------------------------------- #
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate rows from the dataset, if configured to do
        so.

        Args:
            df: DataFrame to deduplicate.

        Returns:
            Deduplicated DataFrame (or the original if disabled).
        """
        if not self.config.drop_duplicates:
            logger.info("Duplicate removal disabled via config; skipping.")
            return df

        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        logger.info("Removed %d duplicate row(s). New shape: %s", removed, df.shape)
        return df

    # ------------------------------------------------------------------- #
    # Step 8: Encode target label
    # ------------------------------------------------------------------- #
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a LabelEncoder on the target column and replace its values
        with integer-encoded classes.

        Args:
            df: DataFrame containing the raw (string) target column.

        Returns:
            DataFrame with the target column encoded as integers.

        Raises:
            DataPreprocessingError: If label encoding fails.
        """
        target = self.config.target_column
        logger.info("Encoding target column: '%s'", target)

        try:
            df[target] = df[target].astype(str).str.strip()
            self.label_encoder = LabelEncoder()
            df[target] = self.label_encoder.fit_transform(df[target])

            mapping = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )
            logger.info("Label encoding mapping: %s", mapping)
            return df

        except (ValueError, TypeError) as exc:
            logger.error("Failed to encode target labels: %s", exc)
            raise DataPreprocessingError(
                f"Failed to encode target labels: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Step 9: Split
    # ------------------------------------------------------------------- #
    def split_features_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the DataFrame into train/test feature matrices and target
        vectors.

        Args:
            df: Fully cleaned and encoded DataFrame.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).

        Raises:
            DataPreprocessingError: If the split operation fails.
        """
        target = self.config.target_column
        logger.info(
            "Splitting data into train/test sets (test_size=%.2f, stratify=%s).",
            self.config.test_size,
            self.config.stratify,
        )

        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Retain only numeric feature columns for modeling.
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                logger.warning(
                    "Dropping non-numeric feature columns prior to split: %s",
                    non_numeric,
                )
                X = X.drop(columns=non_numeric)

            stratify_arg = y if self.config.stratify else None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_arg,
            )

            logger.info(
                "Split complete. X_train: %s, X_test: %s",
                X_train.shape,
                X_test.shape,
            )
            return X_train, X_test, y_train, y_test

        except ValueError as exc:
            logger.error("Train/test split failed: %s", exc)
            raise DataPreprocessingError(f"Train/test split failed: {exc}") from exc

    # ------------------------------------------------------------------- #
    # Step 10: Scale
    # ------------------------------------------------------------------- #
    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit a StandardScaler on the training features and apply it to both
        the training and test sets, preserving column names and indices.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of (X_train_scaled, X_test_scaled) as DataFrames.
        """
        if not self.config.scale_features:
            logger.info("Feature scaling disabled via config; skipping.")
            return X_train, X_test

        logger.info("Scaling numeric features using StandardScaler.")
        try:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            logger.info("Feature scaling complete.")
            return X_train_scaled, X_test_scaled

        except ValueError as exc:
            logger.error("Feature scaling failed: %s", exc)
            raise DataPreprocessingError(f"Feature scaling failed: {exc}") from exc

    # ------------------------------------------------------------------- #
    # Step 11: Persist
    # ------------------------------------------------------------------- #
    def save_artifacts(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Persist processed datasets to the processed data directory and
        fitted transformation artifacts (scaler, label encoder) to the
        models directory.

        Args:
            X_train: Scaled/processed training features.
            X_test: Scaled/processed test features.
            y_train: Encoded training target.
            y_test: Encoded test target.

        Raises:
            DataPreprocessingError: If saving any artifact fails.
        """
        try:
            self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
            self.config.models_dir.mkdir(parents=True, exist_ok=True)

            X_train.to_csv(
                self.config.processed_data_dir / "X_train.csv", index=False
            )
            X_test.to_csv(self.config.processed_data_dir / "X_test.csv", index=False)
            y_train.to_csv(
                self.config.processed_data_dir / "y_train.csv", index=False
            )
            y_test.to_csv(self.config.processed_data_dir / "y_test.csv", index=False)

            pd.Series(X_train.columns, name="feature").to_csv(
                self.config.models_dir / "selected_features.csv", index=False
            )

            if self.scaler is not None:
                joblib.dump(self.scaler, self.config.models_dir / "scaler.pkl")
                logger.info("Scaler artifact saved.")

            if self.label_encoder is not None:
                joblib.dump(
                    self.label_encoder,
                    self.config.models_dir / "label_encoder.pkl",
                )
                logger.info("Label encoder artifact saved.")

            logger.info(
                "All processed datasets and artifacts saved to '%s' and '%s'.",
                self.config.processed_data_dir,
                self.config.models_dir,
            )

        except OSError as exc:
            logger.error("Failed to save preprocessing artifacts: %s", exc)
            raise DataPreprocessingError(
                f"Failed to save preprocessing artifacts: {exc}"
            ) from exc

    # ------------------------------------------------------------------- #
    # Orchestrator
    # ------------------------------------------------------------------- #
    def run(
        self, persist: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full preprocessing pipeline end-to-end.

        Args:
            persist: If True, saves processed datasets and fitted
                artifacts to disk. Set to False for in-memory/testing use.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test) ready for model
            training.

        Raises:
            DataPreprocessingError: Propagated from any pipeline stage on
                unrecoverable failure.
        """
        logger.info("=== Starting data preprocessing pipeline ===")

        df = self.load_data()
        df = self.clean_column_names(df)
        self.validate_data(df)
        df = self.drop_irrelevant_columns(df)
        df = self.handle_missing_and_infinite(df)
        df = self.remove_duplicates(df)
        df = self.encode_labels(df)

        X_train, X_test, y_train, y_test = self.split_features_target(df)
        X_train, X_test = self.scale_features(X_train, X_test)

        if persist:
            self.save_artifacts(X_train, X_test, y_train, y_test)

        logger.info("=== Data preprocessing pipeline completed successfully ===")
        return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Command-line entry point for running the preprocessing pipeline
    standalone, e.g.:

        python -m src.data.data_preprocessing --raw-data data/raw/cicids2017.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess the CICIDS2017 network intrusion dataset."
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Path to the raw CICIDS2017 CSV file or directory of CSV files.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Label",
        help="Name of the target/label column in the dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to reserve for testing.",
    )
    args = parser.parse_args()

    try:
        config = PreprocessingConfig(
            raw_data_path=Path(args.raw_data),
            target_column=args.target_column,
            test_size=args.test_size,
        )
        preprocessor = DataPreprocessor(config)
        preprocessor.run(persist=True)
    except DataPreprocessingError as exc:
        logger.critical("Preprocessing pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()