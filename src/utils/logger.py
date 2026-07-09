"""
logger.py
==========

Centralized logging utility module for the Enterprise Network Intrusion
Intelligence System. Provides a single factory function, :func:`get_logger`,
that all other modules (preprocessing, feature engineering, training,
inference, evaluation, and the dashboard) should use to obtain a
consistently configured logger — with a console handler and, optionally, a
rotating file handler — rather than configuring ``logging`` ad hoc.

This module integrates with :mod:`src.utils.config` when available to
resolve log level, log directory, and rotation settings, but falls back to
sensible defaults if the configuration module cannot be imported or fails
to load, so that logging remains usable even in isolation (e.g., unit
tests, standalone scripts).

Author: Shared Module - Enterprise Network Intrusion Intelligence System
Python Version: 3.11
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Fallback Defaults (used if src.utils.config is unavailable or fails)
# --------------------------------------------------------------------------- #
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_TO_FILE = True
_DEFAULT_LOG_FILENAME = "enterprise_iis.log"
_DEFAULT_LOGS_DIR = Path("logs")
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_DEFAULT_BACKUP_COUNT = 3

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# --------------------------------------------------------------------------- #
# Custom Exceptions
# --------------------------------------------------------------------------- #
class LoggerSetupError(Exception):
    """Raised when logger configuration cannot be completed."""


# --------------------------------------------------------------------------- #
# Internal: Resolve settings from src.utils.config, with graceful fallback
# --------------------------------------------------------------------------- #
def _resolve_logging_settings() -> tuple[str, bool, Path, str, int, int]:
    """
    Resolve effective logging settings, preferring the centralized
    :mod:`src.utils.config` module when it is importable and loads
    successfully, and falling back to module-level defaults otherwise.

    Returns:
        A tuple of (log_level, log_to_file, logs_dir, log_filename,
        max_bytes, backup_count).
    """
    try:
        from src.utils.config import get_config

        app_config = get_config()
        return (
            app_config.logging.log_level.upper(),
            app_config.logging.log_to_file,
            app_config.paths.logs_dir,
            app_config.logging.log_filename,
            app_config.logging.max_bytes,
            app_config.logging.backup_count,
        )
    except Exception:  # noqa: BLE001 - any failure here must not break logging
        return (
            _DEFAULT_LOG_LEVEL,
            _DEFAULT_LOG_TO_FILE,
            _DEFAULT_LOGS_DIR,
            _DEFAULT_LOG_FILENAME,
            _DEFAULT_MAX_BYTES,
            _DEFAULT_BACKUP_COUNT,
        )


# --------------------------------------------------------------------------- #
# Internal: Handler builders
# --------------------------------------------------------------------------- #
def _build_console_handler(level: int) -> logging.StreamHandler:
    """
    Construct a console (stdout) logging handler.

    Args:
        level: Numeric logging level to assign to the handler.

    Returns:
        A configured StreamHandler.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_file_handler(
    logs_dir: Path,
    log_filename: str,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> Optional[RotatingFileHandler]:
    """
    Construct a rotating file logging handler.

    Args:
        logs_dir: Directory in which the log file should be created.
        log_filename: Name of the log file.
        level: Numeric logging level to assign to the handler.
        max_bytes: Maximum size in bytes before the log file rotates.
        backup_count: Number of rotated backup files to retain.

    Returns:
        A configured RotatingFileHandler, or None if the handler could
        not be created (e.g., due to filesystem permission errors). The
        failure is non-fatal so that console logging continues to
        function.
    """
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_filename
        handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        return handler
    except OSError as exc:
        # Fall back to console-only logging rather than raising, since a
        # missing/unwritable log directory should not halt the application.
        fallback_logger = logging.getLogger(__name__)
        fallback_logger.warning(
            "Could not create file handler at '%s/%s': %s. Continuing with "
            "console logging only.",
            logs_dir,
            log_filename,
            exc,
        )
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    logs_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Retrieve (or lazily create) a consistently configured logger for the
    given name.

    On first call for a given ``name``, this attaches a console handler
    and, unless disabled, a rotating file handler to the returned logger,
    then marks it so that subsequent calls do not attach duplicate
    handlers. All explicit arguments override the values resolved from
    :mod:`src.utils.config` (or the module's built-in defaults if that
    config is unavailable).

    Args:
        name: Logger name, conventionally ``__name__`` of the calling
            module.
        level: Optional explicit logging level (e.g., 'DEBUG', 'INFO').
            Overrides the resolved/default level if provided.
        log_to_file: Optional explicit override for whether a rotating
            file handler should be attached.
        logs_dir: Optional explicit override for the directory in which
            log files are written.
        log_filename: Optional explicit override for the log file name.

    Returns:
        A configured ``logging.Logger`` instance ready for use.

    Raises:
        LoggerSetupError: If an invalid ``level`` string is supplied.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline started.")
    """
    if not name:
        raise LoggerSetupError("Logger 'name' must be a non-empty string.")

    (
        resolved_level,
        resolved_log_to_file,
        resolved_logs_dir,
        resolved_log_filename,
        max_bytes,
        backup_count,
    ) = _resolve_logging_settings()

    effective_level_str = (level or resolved_level).upper()
    if effective_level_str not in _VALID_LEVELS:
        raise LoggerSetupError(
            f"Invalid log level '{effective_level_str}'. Must be one of "
            f"{sorted(_VALID_LEVELS)}."
        )
    effective_level = getattr(logging, effective_level_str)

    effective_log_to_file = (
        resolved_log_to_file if log_to_file is None else log_to_file
    )
    effective_logs_dir = logs_dir if logs_dir is not None else resolved_logs_dir
    effective_log_filename = (
        log_filename if log_filename is not None else resolved_log_filename
    )

    logger_instance = logging.getLogger(name)

    # Guard against duplicate handler attachment if get_logger is called
    # multiple times for the same logger name (e.g., across re-imports).
    if getattr(logger_instance, "_en_iis_configured", False):
        logger_instance.setLevel(effective_level)
        return logger_instance

    logger_instance.setLevel(effective_level)
    logger_instance.propagate = False

    logger_instance.addHandler(_build_console_handler(effective_level))

    if effective_log_to_file:
        file_handler = _build_file_handler(
            logs_dir=effective_logs_dir,
            log_filename=effective_log_filename,
            level=effective_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        if file_handler is not None:
            logger_instance.addHandler(file_handler)

    # Mark this logger as fully configured to prevent duplicate handlers.
    logger_instance._en_iis_configured = True  # type: ignore[attr-defined]

    return logger_instance


def set_global_log_level(level: str) -> None:
    """
    Update the logging level for every previously-configured logger
    created via :func:`get_logger`, as well as their attached handlers.
    Useful for dynamically raising verbosity (e.g., a dashboard "debug
    mode" toggle) without recreating loggers.

    Args:
        level: The new logging level to apply (e.g., 'DEBUG', 'WARNING').

    Raises:
        LoggerSetupError: If ``level`` is not a recognized logging level.
    """
    normalized = level.upper()
    if normalized not in _VALID_LEVELS:
        raise LoggerSetupError(
            f"Invalid log level '{normalized}'. Must be one of "
            f"{sorted(_VALID_LEVELS)}."
        )
    numeric_level = getattr(logging, normalized)

    manager = logging.Logger.manager
    for logger_name in manager.loggerDict:  # noqa: WPS528 - intentional registry scan
        candidate = logging.getLogger(logger_name)
        if getattr(candidate, "_en_iis_configured", False):
            candidate.setLevel(numeric_level)
            for handler in candidate.handlers:
                handler.setLevel(numeric_level)


# --------------------------------------------------------------------------- #
# Self-Test Entry Point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Lightweight self-test exercising logger creation, level overrides,
    duplicate-handler prevention, and global level updates.
    """
    demo_logger = get_logger(__name__)
    demo_logger.debug("This DEBUG message should NOT appear at default INFO level.")
    demo_logger.info("Logger initialized successfully.")
    demo_logger.warning("This is a sample warning message.")

    same_logger = get_logger(__name__)
    handler_count_before = len(same_logger.handlers)
    get_logger(__name__)
    handler_count_after = len(same_logger.handlers)
    assert handler_count_before == handler_count_after, "Duplicate handlers attached!"
    demo_logger.info(
        "Duplicate-handler guard verified (%d handler(s) attached).",
        handler_count_after,
    )

    set_global_log_level("DEBUG")
    demo_logger.debug("This DEBUG message SHOULD appear after global level change.")

    demo_logger.info("Self-test completed successfully.")


if __name__ == "__main__":
    main()