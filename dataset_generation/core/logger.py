import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
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

def log_call(verbose: bool = False):
    """
    A decorator to log when a function is called.

    Parameters:
    - logger (logging.Logger): The logger to use.
    - verbose (bool): If True, logs detailed output to the file.

    Returns:
    - function: The wrapped function.
    """ 
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"Function '{func_name}' was called.")
            if verbose:
                logger.debug(f"Function '{func_name}' called with args: {args}, kwargs: {kwargs}")
            print(f"Function '{func_name}' was called.")
            result = func(*args, **kwargs)
            logger.info(f"Function '{func_name}' execution completed.")
            if verbose:
                logger.debug(f"Function '{func_name}' returned: {result}")
            return result
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    logger = setup_logger("my_logger", "app.log", level=logging.DEBUG)

    @log_call(logger, verbose=True)
    def example_function(x, y):
        return x + y

    @log_call(logger, verbose=False)
    def another_function(message):
        print(message)

    # Call the functions
    result = example_function(3, 5)
    another_function("Hello, world!")
