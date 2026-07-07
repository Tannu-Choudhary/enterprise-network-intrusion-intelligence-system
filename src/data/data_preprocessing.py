"""Data preprocessing pipeline for the CIC-IDS2017 network intrusion dataset.

This module defines the ``DataPreprocessor`` class, which implements a full
extract-clean-encode-scale-split-save pipeline for building a network
intrusion intelligence system on top of the CIC-IDS2017 dataset.

Typical usage example:

    preprocessor = DataPreprocessor(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        target_column="Label",
    )
    processed_df = preprocessor.run_pipeline()
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(processed_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


class DataPreprocessor:
    """End-to-end preprocessing pipeline for the CIC-IDS2017 dataset.

    This class loads raw CSV files, inspects and cleans them, encodes
    categorical features, scales numerical features, splits the dataset
    into train/test partitions, and persists the processed dataset to
    disk.

    Attributes:
        raw_data_dir: Directory containing raw CSV files.
        processed_data_dir: Directory where processed artifacts are saved.
        target_column: Name of the target/label column, if present.
        test_size: Fraction of data reserved for the test split.
        random_state: Random seed used for reproducibility.
        raw_data: The merged raw DataFrame after ``load_dataset``.
        processed_data: The cleaned/encoded/scaled DataFrame.
        label_encoders: Mapping of column name to fitted ``LabelEncoder``.
        scaler: Fitted ``StandardScaler`` instance, once ``scale_features``
            has run.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        target_column: Optional[str] = "Label",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Initializes the DataPreprocessor.

        Args:
            raw_data_dir: Path to the folder containing raw CSV files.
            processed_data_dir: Path to the folder where processed data
                will be written.
            target_column: Name of the target column used for stratified
                splitting and to exclude from scaling/encoding-as-feature
                logic. If ``None`` or absent from the data, splitting
                falls back to a non-stratified split.
            test_size: Proportion of the dataset to include in the test
                split.
            random_state: Seed for reproducible results.

        Raises:
            ValueError: If ``test_size`` is not strictly between 0 and 1.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1 (exclusive).")

        self.raw_data_dir: Path = Path(raw_data_dir)
        self.processed_data_dir: Path = Path(processed_data_dir)
        self.target_column: Optional[str] = target_column
        self.test_size: float = test_size
        self.random_state: int = random_state

        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

        logger.info(
            "DataPreprocessor initialized (raw_dir=%s, processed_dir=%s, "
            "target_column=%s, test_size=%.2f, random_state=%d)",
            self.raw_data_dir,
            self.processed_data_dir,
            self.target_column,
            self.test_size,
            self.random_state,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_dataset(self) -> pd.DataFrame:
        """Loads and merges every CSV file found under ``raw_data_dir``.

        Empty files are skipped. Corrupted or unreadable files are logged
        and skipped rather than raising, so one bad file does not abort
        the whole load.

        Returns:
            The merged raw DataFrame, also stored in ``self.raw_data``.

        Raises:
            FileNotFoundError: If ``raw_data_dir`` does not exist.
            ValueError: If no valid CSV files could be loaded.
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}"
            )

        csv_paths: List[Path] = sorted(self.raw_data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in raw data directory: {self.raw_data_dir}"
            )

        loaded_frames: List[pd.DataFrame] = []
        loaded_files: List[str] = []

        for csv_path in csv_paths:
            try:
                if csv_path.stat().st_size == 0:
                    logger.warning("Skipping empty file: %s", csv_path.name)
                    continue

                frame = pd.read_csv(
                    csv_path,
                    low_memory=False,
                    skipinitialspace=True,
                    encoding="utf-8",
                    on_bad_lines="warn",
                )

                if frame.empty:
                    logger.warning(
                        "Skipping file with no rows: %s", csv_path.name
                    )
                    continue

                loaded_frames.append(frame)
                loaded_files.append(csv_path.name)

            except pd.errors.EmptyDataError:
                logger.warning("Skipping empty/unparseable file: %s", csv_path.name)
                continue
            except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
                logger.error("Failed to load corrupted file %s: %s", csv_path.name, exc)
                continue

        if not loaded_frames:
            raise ValueError(
                f"No valid, non-empty CSV files could be loaded from "
                f"{self.raw_data_dir}"
            )

        try:
            merged = pd.concat(loaded_frames, axis=0, ignore_index=True, sort=False)
        except (ValueError, MemoryError) as exc:
            raise ValueError(f"Failed to merge loaded CSV files: {exc}") from exc

        self.raw_data = merged

        logger.info("Dataset loaded")
        logger.info("Files loaded (%d): %s", len(loaded_files), loaded_files)
        logger.info("Dataset shape: %s", self.raw_data.shape)

        return self.raw_data

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def inspect_data(self) -> Dict[str, object]:
        """Produces a summary report describing the loaded raw dataset.

        Returns:
            A dictionary with keys: ``shape``, ``columns``, ``dtypes``,
            ``missing_values``, ``duplicate_count``, and
            ``descriptive_statistics``.

        Raises:
            RuntimeError: If ``load_dataset`` has not been called yet.
        """
        if self.raw_data is None:
            raise RuntimeError(
                "No dataset loaded. Call load_dataset() before inspect_data()."
            )

        data = self.raw_data

        report: Dict[str, object] = {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_count": int(data.duplicated().sum()),
            "descriptive_statistics": data.describe(include="all").to_dict(),
        }

        logger.info("Data inspection complete: shape=%s, duplicates=%d",
                     report["shape"], report["duplicate_count"])

        return report

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean_data(self) -> pd.DataFrame:
        """Cleans the raw dataset in place and stores the result.

        Steps performed:
            1. Strip whitespace from column names.
            2. Remove completely empty columns.
            3. Remove duplicate rows.
            4. Replace +/-Infinity values with NaN.
            5. Drop rows containing any NaN values.

        Returns:
            The cleaned DataFrame, also stored in ``self.processed_data``.

        Raises:
            RuntimeError: If ``load_dataset`` has not been called yet.
        """
        if self.raw_data is None:
            raise RuntimeError(
                "No dataset loaded. Call load_dataset() before clean_data()."
            )

        logger.info("Cleaning started")
        data = self.raw_data.copy()

        # Strip whitespace from column names.
        data.columns = [str(col).strip() for col in data.columns]
        logger.info("Column names stripped of surrounding whitespace")

        # Remove completely empty columns.
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            data = data.drop(columns=empty_columns)
            logger.info("Removed %d completely empty columns: %s",
                        len(empty_columns), empty_columns)
        else:
            logger.info("No completely empty columns found")

        # Remove duplicate rows.
        duplicates_before = int(data.duplicated().sum())
        data = data.drop_duplicates()
        logger.info("Removed %d duplicate rows", duplicates_before)

        # Replace Infinity / -Infinity with NaN.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_mask = data[numeric_cols].isin([np.inf, -np.inf])
        inf_count = int(inf_mask.sum().sum())
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced %d Infinity/-Infinity values with NaN", inf_count)

        # Remove rows containing NaN.
        rows_before = len(data)
        data = data.dropna(axis=0, how="any")
        rows_dropped = rows_before - len(data)
        logger.info("Dropped %d rows containing NaN values", rows_dropped)

        data = data.reset_index(drop=True)
        self.processed_data = data

        logger.info("Cleaning completed. Final shape: %s", data.shape)

        return self.processed_data

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_features(self) -> pd.DataFrame:
        """Label-encodes categorical (non-numeric) feature columns.

        The target column is skipped if it is already numeric. If the
        target column is non-numeric, it is also label-encoded so that it
        can be used for stratified splitting and model training, but its
        encoder is still tracked separately in ``self.label_encoders``.

        Returns:
            The encoded DataFrame, stored in ``self.processed_data``.

        Raises:
            RuntimeError: If ``clean_data`` has not been called yet.
        """
        if self.processed_data is None:
            raise RuntimeError(
                "No cleaned dataset available. Call clean_data() before "
                "encode_features()."
            )

        data = self.processed_data.copy()

        categorical_columns = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_columns:
            logger.info("No categorical columns detected; skipping encoding")
            self.processed_data = data
            logger.info("Encoding completed")
            return self.processed_data

        for column in categorical_columns:
            try:
                encoder = LabelEncoder()
                data[column] = encoder.fit_transform(data[column].astype(str))
                self.label_encoders[column] = encoder
                logger.info(
                    "Encoded categorical column '%s' (%d classes)",
                    column,
                    len(encoder.classes_),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Failed to encode categorical column '{column}': {exc}"
                ) from exc

        self.processed_data = data
        logger.info("Encoding completed")

        return self.processed_data

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def scale_features(self) -> pd.DataFrame:
        """Standard-scales numerical feature columns (excludes the target).

        Returns:
            The scaled DataFrame, stored in ``self.processed_data``.

        Raises:
            RuntimeError: If ``encode_features`` has not been called yet.
        """
        if self.processed_data is None:
            raise RuntimeError(
                "No encoded dataset available. Call encode_features() "
                "before scale_features()."
            )

        data = self.processed_data.copy()

        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)

        if not numerical_columns:
            logger.info("No numerical feature columns detected; skipping scaling")
            self.processed_data = data
            logger.info("Scaling completed")
            return self.processed_data

        try:
            self.scaler = StandardScaler()
            data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])
        except ValueError as exc:
            raise ValueError(f"Failed to scale numerical features: {exc}") from exc

        logger.info(
            "Scaled %d numerical feature columns using StandardScaler",
            len(numerical_columns),
        )

        self.processed_data = data
        logger.info("Scaling completed")

        return self.processed_data

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def split_dataset(
        self, data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the processed dataset into train/test feature and label sets.

        Args:
            data: DataFrame to split. Defaults to ``self.processed_data``
                if not provided.

        Returns:
            A tuple of ``(X_train, X_test, y_train, y_test)``.

        Raises:
            RuntimeError: If no data is available to split.
            ValueError: If ``target_column`` is not present in the data.
        """
        frame = data if data is not None else self.processed_data
        if frame is None:
            raise RuntimeError(
                "No processed dataset available. Run the pipeline before "
                "split_dataset()."
            )

        if self.target_column is None or self.target_column not in frame.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset. "
                f"Available columns: {list(frame.columns)}"
            )

        X = frame.drop(columns=[self.target_column])
        y = frame[self.target_column]

        stratify = y if y.nunique() > 1 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            logger.warning(
                "Stratified split failed (%s); falling back to non-stratified split",
                exc,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        logger.info(
            "Dataset split complete: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_processed_data(self, filename: str = "processed_dataset.csv") -> Path:
        """Saves the processed dataset to ``processed_data_dir``.

        Creates the output directory automatically if it does not exist.

        Args:
            filename: Name of the output CSV file.

        Returns:
            The full path to the saved file.

        Raises:
            RuntimeError: If there is no processed data to save.
            OSError: If the file could not be written.
        """
        if self.processed_data is None:
            raise RuntimeError(
                "No processed dataset available. Run the pipeline before "
                "save_processed_data()."
            )

        try:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Failed to create processed data directory "
                f"{self.processed_data_dir}: {exc}"
            ) from exc

        output_path = self.processed_data_dir / filename

        try:
            self.processed_data.to_csv(output_path, index=False)
        except OSError as exc:
            raise OSError(f"Failed to save processed dataset: {exc}") from exc

        logger.info("Saving completed. Processed data written to: %s", output_path)

        return output_path

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self) -> pd.DataFrame:
        """Runs the full preprocessing pipeline end to end.

        Order: load_dataset -> inspect_data -> clean_data ->
        encode_features -> scale_features -> save_processed_data.

        Returns:
            The final processed DataFrame.

        Raises:
            Exception: Propagates any exception raised by pipeline steps,
                after logging the failure.
        """
        try:
            self.load_dataset()
            self.inspect_data()
            self.clean_data()
            self.encode_features()
            self.scale_features()
            self.save_processed_data()

            logger.info("Pipeline completed successfully")

            return self.processed_data

        except Exception:
            logger.exception("Pipeline execution failed")
            raise


if __name__ == "__main__":
    pipeline = DataPreprocessor(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        target_column="Label",
    )
    final_df = pipeline.run_pipeline()
    X_train, X_test, y_train, y_test = pipeline.split_dataset(final_df)