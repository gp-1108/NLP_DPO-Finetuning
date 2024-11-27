import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(
    logger_name: str,
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,  # Number of backup log files
) -> logging.Logger:
    """
    Set up a logger with a rotating file handler.

    Parameters:
    - logger_name (str): Name of the logger.
    - log_file (str): Path to the log file.
    - level (int): Logging level (default: logging.INFO).
    - max_bytes (int): Maximum size of a log file in bytes before rotation (default: 5 MB).
    - backup_count (int): Number of backup log files to keep (default: 5).

    Returns:
    - logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create rotating file handler
    handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger

# Creating the logger
if "LOG_FILE_PATH" in os.environ:
    logger = setup_logger("LOGGER", os.environ["LOG_FILE_PATH"])
else:
    logger = setup_logger("LOGGER", "app.log")

# Example usage
if __name__ == "__main__":
    logger = setup_logger("my_logger", "app.log", level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
