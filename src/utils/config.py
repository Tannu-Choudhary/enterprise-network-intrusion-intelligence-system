"""
config.py
==========

Centralized configuration module for the Enterprise Network Intrusion
Intelligence System. This module defines all project-wide paths,
hyperparameters, and runtime settings as immutable, strongly-typed
dataclasses, with support for environment-variable overrides. All other
modules (preprocessing, feature engineering, training, inference, and the
dashboard) should import their configuration from this single source of
truth rather than hardcoding paths or constants.

Author: Shared Module - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

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
class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


# --------------------------------------------------------------------------- #
# Environment Variable Helpers
# --------------------------------------------------------------------------- #
def _env_str(key: str, default: str) -> str:
    """
    Read a string value from an environment variable, falling back to a
    default if unset or empty.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is unset or empty.

    Returns:
        The resolved string value.
    """
    value = os.environ.get(key, "").strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    """
    Read an integer value from an environment variable, falling back to a
    default if unset or invalid.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is unset or cannot be parsed.

    Returns:
        The resolved integer value.
    """
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Environment variable '%s'='%s' is not a valid integer; using "
            "default %s.",
            key,
            raw,
            default,
        )
        return default


def _env_float(key: str, default: float) -> float:
    """
    Read a float value from an environment variable, falling back to a
    default if unset or invalid.

    Args:
        key: Environment variable name.
        default: Value to use if the variable is unset or cannot be parsed.

    Returns:
        The resolved float value.
    """
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Environment variable '%s'='%s' is not a valid float; using "
            "default %s.",
            key,
            raw,
            default,
        )
        return default


def _env_bool(key: str, default: bool) -> bool:
    """
    Read a boolean value from an environment variable, falling back to a
    default if unset or unrecognized. Accepts common truthy/falsy string
    representations (case-insensitive).

    Args:
        key: Environment variable name.
        default: Value to use if the variable is unset or unrecognized.

    Returns:
        The resolved boolean value.
    """
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "Environment variable '%s'='%s' is not a recognized boolean; using "
        "default %s.",
        key,
        raw,
        default,
    )
    return default


# --------------------------------------------------------------------------- #
# Project Root Resolution
# --------------------------------------------------------------------------- #
def _resolve_project_root() -> Path:
    """
    Resolve the project's root directory.

    The root is determined by walking up from this file's location
    (src/utils/config.py -> project root is two levels up) unless
    overridden by the ``EN_IIS_PROJECT_ROOT`` environment variable.

    Returns:
        The resolved absolute Path to the project root.
    """
    override = os.environ.get("EN_IIS_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _resolve_project_root()


# --------------------------------------------------------------------------- #
# Path Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathConfig:
    """
    Centralized filesystem paths used throughout the project.

    Attributes:
        project_root: Absolute path to the project root directory.
        data_dir: Root data directory.
        raw_data_dir: Directory containing raw, unmodified dataset files.
        processed_data_dir: Directory containing cleaned/engineered
            datasets produced by the preprocessing and feature engineering
            pipelines.
        models_dir: Directory containing serialized model artifacts
            (best_model.pkl, scaler.pkl, label_encoder.pkl,
            selected_features.csv).
        reports_dir: Root reports directory.
        figures_dir: Directory for saved plot/figure images.
        metrics_dir: Directory for saved evaluation metric files.
        screenshots_dir: Directory for dashboard/application screenshots.
        docs_dir: Root documentation directory.
        logs_dir: Directory where application/pipeline log files are
            written.
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    metrics_dir: Path = PROJECT_ROOT / "reports" / "metrics"
    screenshots_dir: Path = PROJECT_ROOT / "reports" / "screenshots"
    docs_dir: Path = PROJECT_ROOT / "docs"
    logs_dir: Path = PROJECT_ROOT / "logs"

    def ensure_directories_exist(self) -> None:
        """
        Create all configured directories if they do not already exist.

        Raises:
            ConfigError: If a directory cannot be created due to a
                filesystem error (e.g., permissions).
        """
        directories = (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.metrics_dir,
            self.screenshots_dir,
            self.docs_dir,
            self.logs_dir,
        )
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.error("Failed to create directory '%s': %s", directory, exc)
                raise ConfigError(
                    f"Failed to create directory '{directory}': {exc}"
                ) from exc
        logger.debug("All configured project directories verified/created.")


# --------------------------------------------------------------------------- #
# Data Processing Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    """
    Configuration governing data loading, cleaning, and splitting
    behavior, shared between preprocessing and feature engineering
    modules.

    Attributes:
        target_column: Name of the target/label column in the dataset.
        test_size: Fraction of data reserved for the test split.
        random_state: Global random seed for reproducibility.
        drop_duplicates: Whether duplicate rows should be removed during
            cleaning.
        stratify: Whether train/test splits should be stratified on the
            target column.
    """

    target_column: str = field(
        default_factory=lambda: _env_str("EN_IIS_TARGET_COLUMN", "Label")
    )
    test_size: float = field(
        default_factory=lambda: _env_float("EN_IIS_TEST_SIZE", 0.2)
    )
    random_state: int = field(
        default_factory=lambda: _env_int("EN_IIS_RANDOM_STATE", 42)
    )
    drop_duplicates: bool = field(
        default_factory=lambda: _env_bool("EN_IIS_DROP_DUPLICATES", True)
    )
    stratify: bool = field(
        default_factory=lambda: _env_bool("EN_IIS_STRATIFY", True)
    )

    def validate(self) -> None:
        """
        Validate that data configuration values are within acceptable
        ranges.

        Raises:
            ConfigError: If any value is out of its valid range.
        """
        if not 0.0 < self.test_size < 1.0:
            raise ConfigError(
                f"test_size must be between 0 and 1 (exclusive), got "
                f"{self.test_size}."
            )
        if not self.target_column:
            raise ConfigError("target_column must not be empty.")


# --------------------------------------------------------------------------- #
# Model Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration governing model training and evaluation behavior,
    shared between the training, evaluation, and inference modules.

    Attributes:
        model_filename: Filename used to persist/load the best trained
            model artifact.
        scaler_filename: Filename used to persist/load the fitted feature
            scaler.
        label_encoder_filename: Filename used to persist/load the fitted
            target label encoder.
        selected_features_filename: Filename used to persist/load the
            final selected feature list.
        n_estimators: Default number of estimators for ensemble models.
        cv_folds: Default number of cross-validation folds used during
            model selection/evaluation.
        scoring_metric: Default scoring metric used for model selection
            (e.g., 'f1_weighted', 'accuracy', 'roc_auc').
        n_jobs: Number of parallel jobs to use for model fitting where
            supported (-1 uses all available processors).
    """

    model_filename: str = "best_model.pkl"
    scaler_filename: str = "scaler.pkl"
    label_encoder_filename: str = "label_encoder.pkl"
    selected_features_filename: str = "selected_features.csv"
    n_estimators: int = field(
        default_factory=lambda: _env_int("EN_IIS_N_ESTIMATORS", 100)
    )
    cv_folds: int = field(default_factory=lambda: _env_int("EN_IIS_CV_FOLDS", 5))
    scoring_metric: str = field(
        default_factory=lambda: _env_str("EN_IIS_SCORING_METRIC", "f1_weighted")
    )
    n_jobs: int = field(default_factory=lambda: _env_int("EN_IIS_N_JOBS", -1))

    def validate(self) -> None:
        """
        Validate that model configuration values are within acceptable
        ranges.

        Raises:
            ConfigError: If any value is out of its valid range.
        """
        if self.n_estimators <= 0:
            raise ConfigError(
                f"n_estimators must be positive, got {self.n_estimators}."
            )
        if self.cv_folds < 2:
            raise ConfigError(f"cv_folds must be at least 2, got {self.cv_folds}.")


# --------------------------------------------------------------------------- #
# Dashboard Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashboardConfig:
    """
    Configuration governing the Streamlit dashboard application.

    Attributes:
        page_title: Browser tab / page title for the dashboard.
        page_icon: Emoji or icon identifier used as the page favicon.
        layout: Streamlit page layout mode ('wide' or 'centered').
        max_upload_size_mb: Maximum allowed size (in megabytes) for files
            uploaded through the dashboard's prediction interface.
        theme_primary_color: Primary accent color used in custom dashboard
            styling.
    """

    page_title: str = "Enterprise Network Intrusion Intelligence System"
    page_icon: str = ":shield:"
    layout: str = "wide"
    max_upload_size_mb: int = field(
        default_factory=lambda: _env_int("EN_IIS_MAX_UPLOAD_MB", 200)
    )
    theme_primary_color: str = "#0E76A8"


# --------------------------------------------------------------------------- #
# Logging Configuration Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoggingConfig:
    """
    Configuration governing project-wide logging behavior, consumed by
    ``src.utils.logger``.

    Attributes:
        log_level: Root logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_to_file: Whether logs should additionally be written to a
            rotating log file under ``PathConfig.logs_dir``.
        log_filename: Filename used for the log file when
            ``log_to_file`` is True.
        max_bytes: Maximum size in bytes of a single log file before
            rotation occurs.
        backup_count: Number of rotated log file backups to retain.
    """

    log_level: str = field(
        default_factory=lambda: _env_str("EN_IIS_LOG_LEVEL", "INFO")
    )
    log_to_file: bool = field(
        default_factory=lambda: _env_bool("EN_IIS_LOG_TO_FILE", True)
    )
    log_filename: str = "enterprise_iis.log"
    max_bytes: int = field(
        default_factory=lambda: _env_int("EN_IIS_LOG_MAX_BYTES", 5 * 1024 * 1024)
    )
    backup_count: int = field(
        default_factory=lambda: _env_int("EN_IIS_LOG_BACKUP_COUNT", 3)
    )

    def validate(self) -> None:
        """
        Validate that the configured log level is a recognized logging
        level name.

        Raises:
            ConfigError: If ``log_level`` is not a valid logging level.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigError(
                f"log_level must be one of {sorted(valid_levels)}, got "
                f"'{self.log_level}'."
            )


# --------------------------------------------------------------------------- #
# Master Configuration Aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    """
    Top-level aggregate configuration object composing all configuration
    sections. This is the single object that application code should
    request via :func:`get_config`.

    Attributes:
        paths: Filesystem path configuration.
        data: Data processing configuration.
        model: Model training/evaluation configuration.
        dashboard: Dashboard application configuration.
        logging: Logging configuration.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """
        Validate all nested configuration sections that expose a
        ``validate`` method.

        Raises:
            ConfigError: If any nested configuration section fails
                validation.
        """
        logger.info("Validating application configuration.")
        self.data.validate()
        self.model.validate()
        self.logging.validate()
        logger.info("Application configuration validated successfully.")

    def summary(self) -> dict:
        """
        Produce a flat, human-readable dictionary summary of the active
        configuration, useful for startup logging and debugging.

        Returns:
            Dictionary mapping "<section>.<field>" to its resolved value.
        """
        result: dict = {}
        for section_field in fields(self):
            section_name = section_field.name
            section_value = getattr(self, section_name)
            for inner_field in fields(section_value):
                key = f"{section_name}.{inner_field.name}"
                result[key] = getattr(section_value, inner_field.name)
        return result


# --------------------------------------------------------------------------- #
# Singleton Accessor
# --------------------------------------------------------------------------- #
_config_instance: Optional[AppConfig] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """
    Retrieve the application-wide configuration singleton, constructing
    and validating it on first access.

    Args:
        force_reload: If True, discards any cached configuration instance
            and rebuilds it from current environment variables.

    Returns:
        The validated, immutable AppConfig instance.

    Raises:
        ConfigError: If configuration construction or validation fails.
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    try:
        logger.info("Building application configuration.")
        config = AppConfig()
        config.validate()
        config.paths.ensure_directories_exist()
        _config_instance = config
        logger.info("Application configuration ready. Project root: %s", PROJECT_ROOT)
        return config

    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any unexpected error as ConfigError
        logger.error("Unexpected error while building configuration: %s", exc)
        raise ConfigError(
            f"Unexpected error while building configuration: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Command-line entry point for inspecting the resolved configuration,
    e.g.:

        python -m src.utils.config
    """
    try:
        config = get_config()
        logger.info("=== Resolved Configuration ===")
        for key, value in config.summary().items():
            logger.info("%s = %s", key, value)
    except ConfigError as exc:
        logger.critical("Configuration failed to load: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()