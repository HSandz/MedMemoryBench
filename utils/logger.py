"""Logging module - unified logging configuration."""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ERROR_MESSAGE_MAX_LENGTH = 1000


def truncate_error_message(error: object, limit: int = ERROR_MESSAGE_MAX_LENGTH) -> str:
    """Return an error message small enough for logs and run artifacts."""
    message = str(error)
    if len(message) <= limit:
        return message
    if limit <= 3:
        return "." * limit
    return message[:limit - 3] + "..."


def format_limited_traceback(error: BaseException) -> str:
    """Format a traceback without allowing an exception message to dominate it."""
    formatted = traceback.TracebackException(
        type(error), error, error.__traceback__, capture_locals=False
    )
    pending = [formatted]
    while pending:
        exception = pending.pop()
        if isinstance(getattr(exception, "_str", None), str):
            exception._str = truncate_error_message(exception._str)
        for nested in (
            getattr(exception, "__cause__", None),
            getattr(exception, "__context__", None),
        ):
            if nested is not None:
                pending.append(nested)
        pending.extend(getattr(exception, "exceptions", None) or [])
    return "".join(formatted.format())


class _ErrorMessageLimitFilter(logging.Filter):
    """Bound messages emitted at ERROR level by configured experiment loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            record.msg = truncate_error_message(record.getMessage())
            record.args = ()
        return True


def setup_logger(
    name: str = "personalization_lab",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console output
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_ErrorMessageLimitFilter())
        logger.addHandler(console_handler)

    # File output
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_ErrorMessageLimitFilter())
        logger.addHandler(file_handler)

    return logger


def get_eval_logger(
    method_name: str,
    dataset_name: str,
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """Get evaluation logger."""
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / (
        log_filename or f"eval_{method_name}_{dataset_name}_{timestamp}.log"
    )

    logger_name = f"eval.{method_name}.{dataset_name}"
    return setup_logger(logger_name, log_file=log_file)


# Global logger
_main_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get global logger."""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logger("personalization_lab")
    return _main_logger
