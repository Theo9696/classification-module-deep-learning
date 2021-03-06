# -*- coding: utf-8 -*-
# open source project
import logging
import os


DEFAULT_CONSOLE_LEVEL = logging.INFO


def _initialize_logger(output_dir):
    _logger = logging.getLogger("custom")
    _logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(DEFAULT_CONSOLE_LEVEL)
    formatter = logging.Formatter("%(levelname)s (%(asctime)s) - %(module)s %(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "w", encoding=None, delay=True)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s %(funcName)s %(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    # create info file handler and set level to info
    handler = logging.FileHandler(os.path.join(output_dir, "info.log"), "w", encoding=None, delay=True)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "debug.log"), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s %(funcName)s %(lineno)d - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    return _logger


_log_path = os.path.join(os.getcwd(), 'logs/')
os.makedirs(_log_path, exist_ok=True)

logger = _initialize_logger(_log_path)
logger.debug("Logger loaded successfully. Logging directory: {}".format(_log_path))


# For testing purposes.
if __name__ == '__main__':
    logger.debug("Debug message.")
    logger.info("Info message.")
    logger.warning("Warning message.")
    logger.error("Error message.")
    logger.exception(TypeError("Test exception."))
