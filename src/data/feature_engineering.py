"""Feature engineering pipeline for the CIC-IDS2017 network intrusion dataset.

This module provides the ``FeatureEngineer`` class, which encapsulates the
full feature engineering workflow used by the Enterprise Network Intrusion
Intelligence System: irrelevant column removal, numerical feature selection,
correlation analysis, redundant feature pruning, tree-based feature
importance ranking, and persistence of the final selected feature set.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class FeatureEngineer:
    """Performs feature engineering for the CIC-IDS2017 intrusion dataset.

    The pipeline removes irrelevant identifier columns, isolates numerical
    features, analyzes and prunes highly correlated features, computes
    feature importance using a Random Forest classifier, selects the top
    features, and persists the final feature list to disk.

    Attributes:
        correlation_threshold: Pearson correlation threshold above which one
            of a pair of features is dropped.
        top_n_features: Number of top-ranked features to retain.
        irrelevant_columns: Column names considered irrelevant and removed
            prior to analysis.
        output_path: Destination path for the saved selected feature list.
        selected_features_: List of final selected feature names, populated
            after ``run_pipeline`` (or the individual steps) has executed.
        correlation_matrix_: Pearson correlation matrix computed during the
            correlation analysis step.
        feature_importances_: Series of feature importances computed by the
            Random Forest classifier.
    """

    DEFAULT_IRRELEVANT_COLUMNS: List[str] = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
    ]

    def __init__(
        self,
        correlation_threshold: float = 0.95,
        top_n_features: int = 30,
        irrelevant_columns: Optional[List[str]] = None,
        output_path: str = "data/processed/selected_features.csv",
        random_state: int = 42,
    ) -> None:
        """Initializes the FeatureEngineer.

        Args:
            correlation_threshold: Pearson correlation threshold (0-1) above
                which one feature of a correlated pair is dropped.
            top_n_features: Number of top important features to select.
            irrelevant_columns: Optional list of column names to drop before
                analysis. Defaults to the standard CIC-IDS2017 identifier
                columns (Flow ID, Source IP, Destination IP, Timestamp).
            output_path: File path where selected feature names are saved.
            random_state: Random seed for reproducibility of the Random
                Forest classifier.

        Raises:
            ValueError: If ``correlation_threshold`` is not within (0, 1] or
                ``top_n_features`` is not a positive integer.
        """
        if not 0.0 < correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be within (0, 1].")
        if top_n_features <= 0:
            raise ValueError("top_n_features must be a positive integer.")

        self.correlation_threshold = correlation_threshold
        self.top_n_features = top_n_features
        self.irrelevant_columns = (
            irrelevant_columns
            if irrelevant_columns is not None
            else list(self.DEFAULT_IRRELEVANT_COLUMNS)
        )
        self.output_path = Path(output_path)
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.feature_importances_: Optional[pd.Series] = None

        logger.info(
            "FeatureEngineer initialized (correlation_threshold=%.2f, "
            "top_n_features=%d, output_path=%s)",
            self.correlation_threshold,
            self.top_n_features,
            self.output_path,
        )

    def remove_irrelevant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes known irrelevant identifier columns from the dataframe.

        Args:
            df: Input dataframe, typically the raw CIC-IDS2017 dataset.

        Returns:
            A new dataframe with irrelevant columns removed, if present.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        columns_to_drop = [col for col in self.irrelevant_columns if col in df.columns]
        if columns_to_drop:
            logger.info("Removing irrelevant columns: %s", columns_to_drop)
            df = df.drop(columns=columns_to_drop)
        else:
            logger.info("No irrelevant columns found to remove.")

        return df

    def select_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects numerical columns from the dataframe.

        Args:
            df: Input dataframe.

        Returns:
            A dataframe containing only numerical columns.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If no numerical columns are found.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        numerical_df = df.select_dtypes(include=[np.number])

        if numerical_df.empty:
            raise ValueError("No numerical columns found in the dataframe.")

        logger.info(
            "Selected %d numerical columns out of %d total columns.",
            numerical_df.shape[1],
            df.shape[1],
        )
        return numerical_df

    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes the Pearson correlation matrix for numerical features.

        Args:
            df: Dataframe containing numerical features.

        Returns:
            The absolute-value Pearson correlation matrix.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If the dataframe is empty.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Cannot compute correlation on an empty dataframe.")

        logger.info("Computing Pearson correlation matrix on %d features.", df.shape[1])
        corr_matrix = df.corr(method="pearson").abs()
        self.correlation_matrix_ = corr_matrix

        return corr_matrix

    def remove_highly_correlated_features(
        self, df: pd.DataFrame, corr_matrix: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Removes one feature from each pair with correlation above threshold.

        Args:
            df: Dataframe containing numerical features.
            corr_matrix: Optional precomputed correlation matrix. If not
                provided, it is computed from ``df``.

        Returns:
            A dataframe with highly correlated features removed.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        if corr_matrix is None:
            corr_matrix = self.correlation_analysis(df)

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.correlation_threshold)
        ]

        logger.info(
            "Dropping %d highly correlated features (threshold=%.2f): %s",
            len(to_drop),
            self.correlation_threshold,
            to_drop,
        )

        reduced_df = df.drop(columns=to_drop, errors="ignore")
        logger.info(
            "Feature count reduced from %d to %d after correlation pruning.",
            df.shape[1],
            reduced_df.shape[1],
        )

        return reduced_df

    def feature_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.Series:
        """Computes feature importance using a RandomForestClassifier.

        Args:
            X: Feature matrix (numerical, cleaned).
            y: Target labels corresponding to ``X``.

        Returns:
            A pandas Series of feature importances indexed by feature name,
            sorted in descending order.

        Raises:
            TypeError: If ``X`` is not a DataFrame or ``y`` is not a Series.
            ValueError: If ``X`` and ``y`` have mismatched lengths, or ``X``
                is empty.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input 'X' must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("Input 'y' must be a pandas Series.")
        if X.empty:
            raise ValueError("Feature matrix 'X' is empty.")
        if len(X) != len(y):
            raise ValueError("'X' and 'y' must have the same number of rows.")

        logger.info(
            "Training RandomForestClassifier for feature importance on %d "
            "samples and %d features.",
            X.shape[0],
            X.shape[1],
        )

        try:
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X, y)
        except Exception as exc:
            logger.error("RandomForestClassifier training failed: %s", exc)
            raise

        importances = pd.Series(
            clf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        self.feature_importances_ = importances
        logger.info("Feature importance computation complete.")

        return importances

    def select_top_features(
        self, importances: Optional[pd.Series] = None
    ) -> List[str]:
        """Selects the top N most important features.

        Args:
            importances: Optional precomputed feature importance Series. If
                not provided, uses ``self.feature_importances_`` computed by
                a prior call to ``feature_importance``.

        Returns:
            A list of the top ``top_n_features`` feature names, ordered by
            descending importance.

        Raises:
            ValueError: If no feature importances are available.
        """
        if importances is None:
            importances = self.feature_importances_

        if importances is None:
            raise ValueError(
                "No feature importances available. Run 'feature_importance' "
                "first or pass 'importances' explicitly."
            )

        n_select = min(self.top_n_features, len(importances))
        top_features = importances.head(n_select).index.tolist()

        self.selected_features_ = top_features
        logger.info("Selected top %d features: %s", n_select, top_features)

        return top_features

    def save_selected_features(
        self, features: Optional[List[str]] = None
    ) -> Path:
        """Saves the selected feature names to a CSV file.

        Args:
            features: Optional list of feature names to save. If not
                provided, uses ``self.selected_features_``.

        Returns:
            The path to the saved CSV file.

        Raises:
            ValueError: If no features are available to save.
            OSError: If the file cannot be written to disk.
        """
        if features is None:
            features = self.selected_features_

        if not features:
            raise ValueError("No selected features available to save.")

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature_name": features}).to_csv(
                self.output_path, index=False
            )
        except OSError as exc:
            logger.error("Failed to save selected features to %s: %s", self.output_path, exc)
            raise

        logger.info("Saved %d selected features to %s.", len(features), self.output_path)
        return self.output_path

    def run_pipeline(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Executes the full feature engineering pipeline end to end.

        Steps:
            1. Remove irrelevant identifier columns.
            2. Select numerical features.
            3. Compute correlation matrix.
            4. Remove highly correlated features.
            5. Compute feature importance via Random Forest.
            6. Select top N important features.
            7. Save selected features to disk.

        Args:
            df: Raw input dataframe including the target column.
            target_column: Name of the column containing class labels.

        Returns:
            A tuple of (final feature dataframe restricted to selected
            features, list of selected feature names).

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If ``target_column`` is not present in ``df``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        logger.info("Starting feature engineering pipeline on dataframe of shape %s.", df.shape)

        y = df[target_column]

        df_clean = self.remove_irrelevant_features(df.drop(columns=[target_column]))
        numerical_df = self.select_numerical_features(df_clean)

        numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        y = y.loc[numerical_df.index]

        corr_matrix = self.correlation_analysis(numerical_df)
        reduced_df = self.remove_highly_correlated_features(numerical_df, corr_matrix)

        importances = self.feature_importance(reduced_df, y)
        top_features = self.select_top_features(importances)

        self.save_selected_features(top_features)

        final_df = reduced_df[top_features]
        logger.info("Feature engineering pipeline complete. Final shape: %s.", final_df.shape)

        final_df[target_column] = y return final_df, top_features