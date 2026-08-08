import logging
import os
from typing import Optional

# Base directory for storing logs (if not specified through environment variable, set it to `logs` dir under project root)
LOG_DIR = os.getenv("LOG_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "logs"
)
# LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Logging level project-wide
LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


def get_logger(name: str, log_file: Optional[str] = None, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Get a logger with a specific name and optional file logging.

    Args:
        name (str): Logger name, typically the module's `__name__`.
        log_file (str): Log file name. If None, defaults to "<n>.log" under the logs directory.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    # Check if logger already has handlers to avoid duplicate handlers
    if logger.hasHandlers():
        return logger  # Avoid adding multiple handlers to the same logger

    # Prevent propagation to root logger to avoid duplicate messages from basicConfig
    logger.propagate = False

    # Disable propagation to parent loggers to prevent duplicate messages
    logger.setLevel(level)

    # Default to a log file based on the logger name
    # Strip 'src.' prefix if present to avoid 'src/logs' vs 'logs' issues
    clean_name = name.replace("src.", "", 1) if name.startswith("src.") else name
    log_file = log_file or f"{clean_name.replace('.', '->')}.log"
    log_path = os.path.join(LOG_DIR, log_file)

    # Set up file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
