"""
Enterprise Network Intrusion Intelligence System
--------------------------------------------------
Model training module for the CIC-IDS2017 intrusion detection pipeline.

This module defines the ModelTrainer class, which is responsible for:
    - Loading the processed dataset and selected feature list.
    - Preparing training and testing splits.
    - Training Logistic Regression and Random Forest classifiers.
    - Evaluating both models on standard classification metrics.
    - Selecting the best-performing model based on F1-score.
    - Persisting the best model and metrics report to disk.

Author: Senior ML Engineering Team
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, evaluates, and persists intrusion detection models.

    This class orchestrates the end-to-end supervised learning workflow
    for the CIC-IDS2017 network intrusion dataset. It loads a processed
    dataset and a curated list of selected features, trains multiple
    classifiers, evaluates them against standard metrics, and saves the
    best-performing model to disk based on F1-score.

    Attributes:
        dataset_path (Path): Path to the processed dataset CSV file.
        features_path (Path): Path to the selected features CSV file.
        target_column (str): Name of the target/label column.
        model_output_path (Path): Path where the best model is saved.
        metrics_output_path (Path): Path where evaluation metrics are saved.
        test_size (float): Proportion of data reserved for testing.
        random_state (int): Random seed for reproducibility.
        dataset (pd.DataFrame): Loaded processed dataset.
        selected_features (List[str]): List of selected feature names.
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training target vector.
        y_test (pd.Series): Testing target vector.
        trained_models (Dict[str, Any]): Mapping of model name to fitted
            estimator.
        evaluation_results (Dict[str, Dict[str, Any]]): Mapping of model
            name to its computed evaluation metrics.
    """

    def __init__(
        self,
        dataset_path: str = "data/processed/processed_dataset.csv",
        features_path: str = "data/processed/selected_features.csv",
        target_column: str = "Label",
        model_output_path: str = "models/best_model.pkl",
        metrics_output_path: str = "models/model_metrics.csv",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Initializes the ModelTrainer with configuration paths and parameters.

        Args:
            dataset_path: Path to the processed dataset CSV file.
            features_path: Path to the CSV file containing selected feature
                names.
            target_column: Name of the label/target column in the dataset.
            model_output_path: Destination path for the serialized best model.
            metrics_output_path: Destination path for the metrics report CSV.
            test_size: Fraction of the dataset to reserve for testing.
            random_state: Seed used for reproducible train/test splits and
                model initialization.
        """
        self.dataset_path = Path(dataset_path)
        self.features_path = Path(features_path)
        self.target_column = target_column
        self.model_output_path = Path(model_output_path)
        self.metrics_output_path = Path(metrics_output_path)
        self.test_size = test_size
        self.random_state = random_state

        self.dataset: pd.DataFrame = pd.DataFrame()
        self.selected_features: List[str] = []

        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()
        self.y_train: pd.Series = pd.Series(dtype="object")
        self.y_test: pd.Series = pd.Series(dtype="object")

        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}

        logger.info("ModelTrainer initialized successfully.")

    # ----------------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------------
    def load_dataset(self) -> pd.DataFrame:
        """Loads the processed dataset from disk.

        Returns:
            The loaded dataset as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is empty or the target column is
                missing/invalid.
        """
        logger.info("Loading dataset from '%s'.", self.dataset_path)

        if not self.dataset_path.exists():
            logger.error("Dataset file not found at '%s'.", self.dataset_path)
            raise FileNotFoundError(
                f"Dataset file not found at '{self.dataset_path}'. "
                "Ensure the processed dataset has been generated."
            )

        try:
            self.dataset = pd.read_csv(self.dataset_path)
        except Exception as exc:  # noqa: BLE001 - surface any parsing error
            logger.error("Failed to read dataset: %s", exc)
            raise ValueError(f"Unable to parse dataset CSV: {exc}") from exc

        if self.dataset.empty:
            logger.error("Loaded dataset is empty.")
            raise ValueError("Loaded dataset is empty. Cannot proceed with training.")

        if self.target_column not in self.dataset.columns:
            logger.error(
                "Target column '%s' not found in dataset columns.",
                self.target_column,
            )
            raise ValueError(
                f"Target column '{self.target_column}' is missing from the "
                "dataset. Cannot proceed with training."
            )

        if self.dataset[self.target_column].nunique(dropna=True) < 2:
            logger.error("Target column '%s' has fewer than 2 classes.", self.target_column)
            raise ValueError(
                f"Target column '{self.target_column}' must contain at least "
                "two distinct classes for classification."
            )

        logger.info(
            "Dataset loaded successfully with shape %s.", self.dataset.shape
        )
        return self.dataset

    def load_selected_features(self) -> List[str]:
        """Loads the list of selected feature names from disk.

        The selected features CSV is expected to contain a single column
        of feature names (header optional, but recommended).

        Returns:
            A list of selected feature column names.

        Raises:
            FileNotFoundError: If the selected features file does not exist.
            ValueError: If the file is empty or contains no valid features
                present in the dataset.
        """
        logger.info(
            "Loading selected features from '%s'.", self.features_path
        )

        if not self.features_path.exists():
            logger.error(
                "Selected features file not found at '%s'.", self.features_path
            )
            raise FileNotFoundError(
                f"Selected features file not found at '{self.features_path}'."
            )

        try:
            features_df = pd.read_csv(self.features_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read selected features file: %s", exc)
            raise ValueError(
                f"Unable to parse selected features CSV: {exc}"
            ) from exc

        if features_df.empty:
            logger.error("Selected features file is empty.")
            raise ValueError("Selected features file contains no data.")

        # Use the first column regardless of its header name.
        feature_list = (
            features_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        )

        if not feature_list:
            logger.error("No valid feature names found in selected features file.")
            raise ValueError("Selected features file contains no valid feature names.")

        # Validate features exist within the loaded dataset (if already loaded).
        if not self.dataset.empty:
            missing = [f for f in feature_list if f not in self.dataset.columns]
            if missing:
                logger.warning(
                    "The following selected features are missing from the "
                    "dataset and will be ignored: %s",
                    missing,
                )
                feature_list = [f for f in feature_list if f in self.dataset.columns]

            if not feature_list:
                logger.error(
                    "None of the selected features exist in the dataset."
                )
                raise ValueError(
                    "None of the selected features are present in the dataset."
                )

        self.selected_features = feature_list
        logger.info(
            "Loaded %d selected feature(s).", len(self.selected_features)
        )
        return self.selected_features

    # ----------------------------------------------------------------------
    # Data preparation
    # ----------------------------------------------------------------------
    def prepare_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepares training and testing splits using the selected features.

        Returns:
            A tuple of (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: If the dataset or selected features have not been
                loaded, or if data preparation fails.
        """
        if self.dataset.empty:
            logger.error("Dataset has not been loaded prior to preparation.")
            raise ValueError("Dataset must be loaded before preparing training data.")

        if not self.selected_features:
            logger.error("Selected features have not been loaded prior to preparation.")
            raise ValueError(
                "Selected features must be loaded before preparing training data."
            )

        logger.info("Preparing training and testing datasets.")

        try:
            features_df = self.dataset[self.selected_features].copy()
            target_series = self.dataset[self.target_column].copy()

            # Drop rows with missing values to ensure model stability.
            combined = pd.concat([features_df, target_series], axis=1)
            initial_rows = len(combined)
            combined = combined.dropna()
            dropped_rows = initial_rows - len(combined)
            if dropped_rows > 0:
                logger.warning(
                    "Dropped %d row(s) containing missing values.", dropped_rows
                )

            if combined.empty:
                raise ValueError(
                    "No data remains after dropping missing values."
                )

            features_df = combined[self.selected_features]
            target_series = combined[self.target_column]

            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                features_df,
                target_series,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=target_series,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to prepare training data: %s", exc)
            raise ValueError(f"Data preparation failed: {exc}") from exc

        logger.info(
            "Training data prepared. Train shape: %s | Test shape: %s.",
            self.X_train.shape,
            self.X_test.shape,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ----------------------------------------------------------------------
    # Model training
    # ----------------------------------------------------------------------
    def train_logistic_regression(self) -> LogisticRegression:
        """Trains a Logistic Regression classifier on the prepared data.

        Returns:
            The fitted LogisticRegression model.

        Raises:
            ValueError: If training data has not been prepared.
            RuntimeError: If model training fails.
        """
        self._validate_training_data_ready()
        logger.info("Training Logistic Regression model.")

        try:
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(self.X_train, self.y_train)
        except Exception as exc:  # noqa: BLE001
            logger.error("Logistic Regression training failed: %s", exc)
            raise RuntimeError(
                f"Logistic Regression training failed: {exc}"
            ) from exc

        self.trained_models["LogisticRegression"] = model
        logger.info("Logistic Regression training complete.")
        return model

    def train_random_forest(self) -> RandomForestClassifier:
        """Trains a Random Forest classifier on the prepared data.

        Returns:
            The fitted RandomForestClassifier model.

        Raises:
            ValueError: If training data has not been prepared.
            RuntimeError: If model training fails.
        """
        self._validate_training_data_ready()
        logger.info("Training Random Forest model.")

        try:
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(self.X_train, self.y_train)
        except Exception as exc:  # noqa: BLE001
            logger.error("Random Forest training failed: %s", exc)
            raise RuntimeError(f"Random Forest training failed: {exc}") from exc

        self.trained_models["RandomForest"] = model
        logger.info("Random Forest training complete.")
        return model

    def _validate_training_data_ready(self) -> None:
        """Validates that training data has been prepared.

        Raises:
            ValueError: If X_train or y_train are empty.
        """
        if self.X_train.empty or self.y_train.empty:
            logger.error("Training data is not available.")
            raise ValueError(
                "Training data has not been prepared. Call "
                "prepare_training_data() before training a model."
            )

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """Evaluates all trained models on the test set.

        Computes accuracy, precision, recall, F1-score, confusion matrix,
        and a full classification report for each trained model.

        Returns:
            A dictionary mapping model name to its evaluation metrics.

        Raises:
            ValueError: If no models have been trained yet, or test data
                is unavailable.
        """
        if not self.trained_models:
            logger.error("No trained models available for evaluation.")
            raise ValueError(
                "No models have been trained. Train at least one model "
                "before calling evaluate_models()."
            )

        if self.X_test.empty or self.y_test.empty:
            logger.error("Test data is not available for evaluation.")
            raise ValueError(
                "Test data has not been prepared. Call "
                "prepare_training_data() before evaluation."
            )

        logger.info("Evaluating trained models.")
        results: Dict[str, Dict[str, Any]] = {}

        for model_name, model in self.trained_models.items():
            logger.info("Evaluating model: %s.", model_name)
            try:
                predictions = model.predict(self.X_test)

                accuracy = accuracy_score(self.y_test, predictions)
                precision = precision_score(
                    self.y_test, predictions, average="weighted", zero_division=0
                )
                recall = recall_score(
                    self.y_test, predictions, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    self.y_test, predictions, average="weighted", zero_division=0
                )
                conf_matrix = confusion_matrix(self.y_test, predictions)
                report = classification_report(
                    self.y_test, predictions, zero_division=0
                )

                results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": conf_matrix,
                    "classification_report": report,
                }

                logger.info(
                    "%s -> Accuracy: %.4f | Precision: %.4f | Recall: %.4f | "
                    "F1-score: %.4f",
                    model_name,
                    accuracy,
                    precision,
                    recall,
                    f1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Evaluation failed for model '%s': %s", model_name, exc)
                raise RuntimeError(
                    f"Evaluation failed for model '{model_name}': {exc}"
                ) from exc

        self.evaluation_results = results
        logger.info("Model evaluation complete.")
        return results

    # ----------------------------------------------------------------------
    # Model persistence
    # ----------------------------------------------------------------------
    def save_best_model(self) -> str:
        """Selects the best model by F1-score and persists it to disk.

        Also writes a metrics report CSV summarizing all evaluated models.

        Returns:
            The name of the best-performing model.

        Raises:
            ValueError: If no evaluation results are available.
            RuntimeError: If saving the model or metrics fails.
        """
        if not self.evaluation_results:
            logger.error("No evaluation results available for model selection.")
            raise ValueError(
                "No evaluation results found. Call evaluate_models() before "
                "save_best_model()."
            )

        best_model_name = max(
            self.evaluation_results,
            key=lambda name: self.evaluation_results[name]["f1_score"],
        )
        best_model = self.trained_models[best_model_name]
        best_f1 = self.evaluation_results[best_model_name]["f1_score"]

        logger.info(
            "Best model selected: '%s' with F1-score %.4f.",
            best_model_name,
            best_f1,
        )

        try:
            self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, self.model_output_path)
            logger.info("Best model saved to '%s'.", self.model_output_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save best model: %s", exc)
            raise RuntimeError(f"Failed to save best model: {exc}") from exc

        try:
            self._save_metrics_report()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save metrics report: %s", exc)
            raise RuntimeError(f"Failed to save metrics report: {exc}") from exc

        return best_model_name

    def _save_metrics_report(self) -> None:
        """Writes a CSV summary of evaluation metrics for all models.

        Raises:
            RuntimeError: If the metrics file cannot be written.
        """
        logger.info("Saving metrics report to '%s'.", self.metrics_output_path)

        rows = []
        for model_name, metrics in self.evaluation_results.items():
            rows.append(
                {
                    "model_name": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "classification_report": metrics["classification_report"],
                }
            )

        metrics_df = pd.DataFrame(rows)

        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(self.metrics_output_path, index=False)
        logger.info("Metrics report saved successfully.")

    # ----------------------------------------------------------------------
    # Pipeline orchestration
    # ----------------------------------------------------------------------
    def run_pipeline(self) -> str:
        """Executes the full model training pipeline end-to-end.

        Steps:
            1. Load the processed dataset.
            2. Load the selected feature list.
            3. Prepare training/testing splits.
            4. Train Logistic Regression and Random Forest models.
            5. Evaluate both models.
            6. Save the best model and metrics report.

        Returns:
            The name of the best-performing model.

        Raises:
            Exception: Propagates any exception raised during the pipeline,
                after logging the failure context.
        """
        logger.info("Starting model training pipeline.")

        try:
            self.load_dataset()
            self.load_selected_features()
            self.prepare_training_data()

            self.train_logistic_regression()
            self.train_random_forest()

            self.evaluate_models()
            best_model_name = self.save_best_model()

            logger.info(
                "Pipeline completed successfully. Best model: '%s'.",
                best_model_name,
            )
            return best_model_name
        except Exception as exc:  # noqa: BLE001
            logger.error("Model training pipeline failed: %s", exc)
            raise


def main() -> None:
    """Entry point for running the model training pipeline as a script."""
    trainer = ModelTrainer()
    trainer.run_pipeline()


if __name__ == "__main__":
    main()