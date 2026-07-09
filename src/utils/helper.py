"""
helper.py
==========

Shared general-purpose utility functions for the Enterprise Network
Intrusion Intelligence System. This module collects small, reusable
helpers — timing decorators, artifact serialization wrappers, filesystem
helpers, reproducibility seeding, DataFrame validation, class-weight
computation, and human-readable formatting — that are used across the
preprocessing, feature engineering, training, inference, evaluation, and
dashboard modules to avoid duplicating boilerplate logic.

Author: Shared Module - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import functools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logger Resolution (prefer shared logger; fall back to stdlib logging)
# --------------------------------------------------------------------------- #
try:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:  # noqa: BLE001 - logger module may be unavailable in isolation
    import logging

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

F = TypeVar("F", bound=Callable[..., Any])


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class HelperError(Exception):
    """Raised when a shared helper utility encounters an unrecoverable error."""


# --------------------------------------------------------------------------- #
# Timing Utilities
# --------------------------------------------------------------------------- #
def timeit(func: F) -> F:
    """
    Decorator that logs the execution time of the wrapped function at INFO
    level upon completion, and re-raises any exception after logging the
    elapsed time at ERROR level.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function with timing instrumentation.

    Example:
        >>> @timeit
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()  # doctest: +SKIP
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("'%s' completed in %.4f seconds.", func.__qualname__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error(
                "'%s' raised an exception after %.4f seconds.",
                func.__qualname__,
                elapsed,
            )
            raise

    return wrapper  # type: ignore[return-value]


class Timer:
    """
    Context manager for timing a block of code, logging the elapsed time
    on exit.

    Example:
        >>> with Timer("data loading"):
        ...     time.sleep(0.1)  # doctest: +SKIP
    """

    def __init__(self, label: str = "operation") -> None:
        """
        Initialize the Timer.

        Args:
            label: Human-readable description of the timed operation,
                used in the logged message.
        """
        self.label = label
        self._start: Optional[float] = None
        self.elapsed_seconds: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.elapsed_seconds = time.perf_counter() - (self._start or time.perf_counter())
        if exc_type is None:
            logger.info("'%s' completed in %.4f seconds.", self.label, self.elapsed_seconds)
        else:
            logger.error(
                "'%s' raised %s after %.4f seconds.",
                self.label,
                exc_type.__name__,
                self.elapsed_seconds,
            )


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_random_seed(seed: int = 42) -> None:
    """
    Seed Python's ``random`` module and NumPy's global RNG for
    reproducibility across preprocessing, feature engineering, and model
    training runs.

    Args:
        seed: The random seed value to apply.

    Raises:
        HelperError: If seeding fails unexpectedly.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        logger.info("Global random seed set to %d.", seed)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to set global random seed: %s", exc)
        raise HelperError(f"Failed to set global random seed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Filesystem Helpers
# --------------------------------------------------------------------------- #
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it (and any missing parents) if
    necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The resolved Path object.

    Raises:
        HelperError: If the directory cannot be created.
    """
    resolved = Path(path)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    except OSError as exc:
        logger.error("Failed to create directory '%s': %s", resolved, exc)
        raise HelperError(f"Failed to create directory '{resolved}': {exc}") from exc


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a formatted timestamp string, commonly used to create unique
    filenames for reports, model checkpoints, or log files.

    Args:
        fmt: ``time.strftime``-compatible format string.

    Returns:
        The formatted current timestamp.
    """
    return time.strftime(fmt)


def format_bytes(num_bytes: float) -> str:
    """
    Convert a byte count into a human-readable string with an appropriate
    unit suffix (B, KB, MB, GB, TB).

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        Human-readable string representation (e.g., "12.34 MB").

    Raises:
        HelperError: If ``num_bytes`` is negative.
    """
    if num_bytes < 0:
        raise HelperError(f"num_bytes must be non-negative, got {num_bytes}.")

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


# --------------------------------------------------------------------------- #
# Artifact Serialization
# --------------------------------------------------------------------------- #
def save_object(obj: Any, path: Union[str, Path]) -> Path:
    """
    Serialize an arbitrary Python object (model, scaler, encoder, etc.)
    to disk using joblib, creating parent directories as needed.

    Args:
        obj: The object to serialize.
        path: Destination file path.

    Returns:
        The resolved Path the object was written to.

    Raises:
        HelperError: If serialization fails.
    """
    resolved = Path(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, resolved)
        logger.info("Saved object of type '%s' to '%s'.", type(obj).__name__, resolved)
        return resolved
    except (OSError, ValueError, TypeError) as exc:
        logger.error("Failed to save object to '%s': %s", resolved, exc)
        raise HelperError(f"Failed to save object to '{resolved}': {exc}") from exc


def load_object(path: Union[str, Path]) -> Any:
    """
    Deserialize a Python object previously saved with :func:`save_object`.

    Args:
        path: Path to the serialized object file.

    Returns:
        The deserialized object.

    Raises:
        HelperError: If the file does not exist or cannot be
            deserialized.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise HelperError(f"Cannot load object: file does not exist at '{resolved}'.")

    try:
        obj = joblib.load(resolved)
        logger.info("Loaded object of type '%s' from '%s'.", type(obj).__name__, resolved)
        return obj
    except (OSError, ValueError, EOFError) as exc:
        logger.error("Failed to load object from '%s': %s", resolved, exc)
        raise HelperError(f"Failed to load object from '{resolved}': {exc}") from exc


def save_json(data: dict, path: Union[str, Path], indent: int = 2) -> Path:
    """
    Persist a dictionary as a formatted JSON file, creating parent
    directories as needed.

    Args:
        data: The dictionary to serialize.
        path: Destination file path.
        indent: Number of spaces used for JSON indentation.

    Returns:
        The resolved Path the JSON was written to.

    Raises:
        HelperError: If serialization fails.
    """
    resolved = Path(path)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info("Saved JSON data to '%s'.", resolved)
        return resolved
    except (OSError, TypeError) as exc:
        logger.error("Failed to save JSON to '%s': %s", resolved, exc)
        raise HelperError(f"Failed to save JSON to '{resolved}': {exc}") from exc


def load_json(path: Union[str, Path]) -> dict:
    """
    Load a dictionary from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialized dictionary.

    Raises:
        HelperError: If the file does not exist or contains invalid JSON.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise HelperError(f"Cannot load JSON: file does not exist at '{resolved}'.")

    try:
        with resolved.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON data from '%s'.", resolved)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load JSON from '%s': %s", resolved, exc)
        raise HelperError(f"Failed to load JSON from '{resolved}': {exc}") from exc


# --------------------------------------------------------------------------- #
# DataFrame Validation
# --------------------------------------------------------------------------- #
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    allow_empty: bool = False,
) -> None:
    """
    Validate structural properties of a DataFrame commonly required
    across pipeline stages: non-null type, non-emptiness, and presence of
    required columns.

    Args:
        df: The DataFrame to validate.
        required_columns: Optional list of column names that must be
            present in ``df``.
        allow_empty: If False, raises an error when ``df`` has zero rows.

    Raises:
        HelperError: If any validation check fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise HelperError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    if df.empty and not allow_empty:
        raise HelperError("DataFrame is empty.")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise HelperError(f"DataFrame is missing required column(s): {missing}")

    logger.debug("DataFrame validation passed. Shape: %s", df.shape)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns of a DataFrame to the smallest safe dtype in
    order to reduce memory footprint, which is particularly useful given
    the large row counts of CICIDS2017 CSV exports.

    Args:
        df: The DataFrame to optimize.

    Returns:
        A memory-optimized copy of the DataFrame.

    Raises:
        HelperError: If downcasting fails unexpectedly.
    """
    try:
        start_mem = df.memory_usage(deep=True).sum()
        df = df.copy()

        for col in df.select_dtypes(include=["int"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        end_mem = df.memory_usage(deep=True).sum()
        reduction_pct = (
            100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0.0
        )
        logger.info(
            "Memory usage reduced from %s to %s (%.1f%% reduction).",
            format_bytes(start_mem),
            format_bytes(end_mem),
            reduction_pct,
        )
        return df

    except (ValueError, TypeError) as exc:
        logger.error("Failed to reduce DataFrame memory usage: %s", exc)
        raise HelperError(f"Failed to reduce DataFrame memory usage: {exc}") from exc


# --------------------------------------------------------------------------- #
# Class Imbalance Helpers
# --------------------------------------------------------------------------- #
def compute_class_weights(y: Union[pd.Series, np.ndarray]) -> dict[Any, float]:
    """
    Compute balanced class weights inversely proportional to class
    frequency, useful for training classifiers on the highly imbalanced
    CICIDS2017 traffic classes (benign traffic vastly outnumbers most
    attack types).

    Args:
        y: Target label vector (encoded or raw).

    Returns:
        Dictionary mapping each unique class label to its computed
        weight.

    Raises:
        HelperError: If ``y`` is empty or contains a single class.
    """
    y_array = np.asarray(y)
    if y_array.size == 0:
        raise HelperError("Cannot compute class weights: target vector is empty.")

    classes, counts = np.unique(y_array, return_counts=True)
    if len(classes) < 2:
        raise HelperError(
            "Cannot compute class weights: target vector contains only one "
            "unique class."
        )

    n_samples = y_array.size
    n_classes = len(classes)
    weights = {
        cls: n_samples / (n_classes * count) for cls, count in zip(classes, counts)
    }

    logger.info("Computed class weights for %d class(es): %s", n_classes, weights)
    return weights


# --------------------------------------------------------------------------- #
# Miscellaneous
# --------------------------------------------------------------------------- #
def chunk_list(items: list, chunk_size: int) -> list[list]:
    """
    Split a list into consecutive chunks of at most ``chunk_size``
    elements, useful for batch-processing large prediction requests in
    the inference/dashboard modules.

    Args:
        items: The list to split.
        chunk_size: Maximum number of elements per chunk.

    Returns:
        A list of list chunks.

    Raises:
        HelperError: If ``chunk_size`` is not a positive integer.
    """
    if chunk_size <= 0:
        raise HelperError(f"chunk_size must be positive, got {chunk_size}.")

    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform division while safely handling a zero denominator, avoiding
    ``ZeroDivisionError`` in metric calculations (e.g., rates computed
    from small or degenerate class counts).

    Args:
        numerator: The division numerator.
        denominator: The division denominator.
        default: Value returned if ``denominator`` is zero.

    Returns:
        The division result, or ``default`` if the denominator is zero.
    """
    if denominator == 0:
        logger.debug(
            "safe_divide: denominator is zero; returning default value %s.", default
        )
        return default
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Self-Test Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Lightweight self-test exercising the primary helper utilities using
    synthetic data.
    """
    set_global_random_seed(42)

    with Timer("synthetic sleep"):
        time.sleep(0.05)

    @timeit
    def _sample_task() -> int:
        return sum(range(1000))

    _sample_task()

    logger.info("format_bytes(1536): %s", format_bytes(1536))
    logger.info("safe_divide(10, 0): %s", safe_divide(10, 0))
    logger.info("chunk_list: %s", chunk_list(list(range(7)), 3))

    df = pd.DataFrame(
        {"a": np.random.randint(0, 100, 1000), "b": np.random.rand(1000)}
    )
    validate_dataframe(df, required_columns=["a", "b"])
    reduce_memory_usage(df)

    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    compute_class_weights(y)

    logger.info("Self-test completed successfully.")


if __name__ == "__main__":
    main()