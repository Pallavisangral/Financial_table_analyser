# Four files of 20 MB will be kept
r""" server.logging module """

# importing standard modules ==================================================
import logging
from logging.handlers import RotatingFileHandler

# class definitions  ==========================================================


class LoggingHandle:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)

        file_handler = RotatingFileHandler(
            "error.log",
            mode="a",
            maxBytes=20 * 1024 * 1024,
            backupCount=4,
            encoding=None,
            delay=0,
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def write_log(self, error_traceback):
        self.logger.error("An error occurred: %s", error_traceback)
