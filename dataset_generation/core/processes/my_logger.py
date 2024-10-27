import logging
import functools
import traceback
from datetime import datetime

class Logger:
    """
    A comprehensive logging class to handle console and file logging
    with different levels of severity.
    """

    def __init__(self, name: str, log_file: str = 'app.log', level=logging.DEBUG):
        """
        Initializes the Logger object with a file handler and console handler.

        Args:
            name (str): Name of the logger.
            log_file (str, optional): File to log messages to. Defaults to 'app.log'.
            level (int, optional): The minimum logging level. Defaults to logging.DEBUG.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter for both file and console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Adding the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg: str):
        """
        Logs a debug message.

        Args:
            msg (str): The message to log.
        """
        self.logger.debug(msg)

    def info(self, msg: str):
        """
        Logs an informational message.

        Args:
            msg (str): The message to log.
        """
        self.logger.info(msg)

    def warning(self, msg: str):
        """
        Logs a warning message.

        Args:
            msg (str): The message to log.
        """
        self.logger.warning(msg)

    def error(self, msg: str):
        """
        Logs an error message.

        Args:
            msg (str): The message to log.
        """
        self.logger.error(msg)

    def critical(self, msg: str):
        """
        Logs a critical message.

        Args:
            msg (str): The message to log.
        """
        self.logger.critical(msg)

    def log_exception(self, msg: str):
        """
        Logs an exception along with its traceback.

        Args:
            msg (str): The message to log along with the exception details.
        """
        self.logger.error(f"{msg}\n{traceback.format_exc()}")

def log_function_call(func, app_logger):
    """
    A decorator to log the execution of a function, its arguments,
    and whether it succeeds or fails.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapped function with logging added.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wraps the original function to log its call, arguments, execution time,
        and success or failure.

        Args:
            *args: Positional arguments passed to the wrapped function.
            **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
            Any: The return value of the wrapped function, if successful.

        Raises:
            Exception: If the wrapped function raises an exception, it is logged and re-raised.
        """
        start_time = datetime.now()
        try:
            # Log the function call and arguments
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            app_logger.info(f"Calling {func.__name__} with arguments: {signature}")

            # Call the actual function
            result = func(*args, **kwargs)

            # Log successful completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            app_logger.info(f"Function {func.__name__} executed successfully in {duration:.4f} seconds")
            
            return result
        except Exception as e:
            # Log the error and exception details
            app_logger.log_exception(f"Function {func.__name__} failed with error: {e}")
            raise
    return wrapper
