# global
import os
import threading
import logging


__all__ = []

VERBOSITY_ENV_NAME = "TRANSLATOR_VERBOSITY"
CODE_LEVEL_ENV_NAME = "TRANSLATOR_CODE_LEVEL"
DEFAULT_VERBOSITY = -1
DEFAULT_CODE_LEVEL = -1
MAX_TRANSFORMERS = 100
USE_LOGGING = __file__.endswith(
    ".py"
)  # don't log anything if binarised  # TODO: check this works


def synchronized(func):
    def wrapper(*args, **kwargs):
        with threading.Lock():
            return func(*args, **kwargs)

    return wrapper


def get_logger(name, level, fmt=None, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if filename:
        handler = logging.FileHandler(filename, mode="w")
    else:
        handler = logging.StreamHandler()

    if fmt:
        formatter = logging.Formatter(fmt=fmt, datefmt="%a %b %d %H:%M:%S")
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_verbosity(level=0, also_to_stdout=False):
    """
    Sets the verbosity level of log for Source2Source. Logs can be output to stdout by setting `also_to_stdout`.
    """
    _TRANSLATOR_LOGGER.verbosity_level = level
    _TRANSLATOR_LOGGER.need_to_echo_log_to_stdout = also_to_stdout


def get_verbosity():
    return _TRANSLATOR_LOGGER.verbosity_level


def set_code_level(level=MAX_TRANSFORMERS, also_to_stdout=False):
    """
    Sets the level to print code from specific level Ast Transformer. Code can be output to stdout by setting `also_to_stdout`.
    """
    _TRANSLATOR_LOGGER.transformed_code_level = level
    _TRANSLATOR_LOGGER.need_to_echo_code_to_stdout = also_to_stdout


def get_code_level():
    return _TRANSLATOR_LOGGER.transformed_code_level


def error(msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.error(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.warn(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    _TRANSLATOR_LOGGER.log(level, msg, *args, **kwargs)


def log_transformed_code(level, ast_node, transformer_name, *args, **kwargs):
    _TRANSLATOR_LOGGER.log_transformed_code(
        level, ast_node, transformer_name, *args, **kwargs
    )


class Logger:
    """
    class for Logging and debugging during the Source2Source transformation.
    The object of this class is a singleton.
    """

    @synchronized
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logger_name = "Source-to-Source"
        self._logger = (
            get_logger(
                self.logger_name,
                1,
                fmt="%(asctime)s %(name)s %(levelname)s: %(message)s",
                filename="ivy_logs.txt",
            )
            if USE_LOGGING
            else None
        )
        self._verbosity_level = None
        self._transformed_code_level = []
        self._need_to_echo_log_to_stdout = None
        self._need_to_echo_code_to_stdout = None

    @property
    def logger(self):
        return self._logger

    @property
    def verbosity_level(self):
        if self._verbosity_level is not None:
            return self._verbosity_level
        else:
            return int(os.getenv(VERBOSITY_ENV_NAME, DEFAULT_VERBOSITY))

    @verbosity_level.setter
    def verbosity_level(self, level):
        self.check_level(level)
        self._verbosity_level = level

    @property
    def transformed_code_level(self):
        if self._transformed_code_level:
            return self._transformed_code_level
        else:
            env_level = os.getenv(CODE_LEVEL_ENV_NAME, DEFAULT_CODE_LEVEL)
            if env_level != -1:
                return list(map(int, env_level.split(",")))
            return [DEFAULT_CODE_LEVEL]

    @transformed_code_level.setter
    def transformed_code_level(self, levels):
        if isinstance(levels, int):
            levels = [levels]
        elif not isinstance(levels, list):
            raise TypeError(f"Level is not an integer or list of integers: {levels}")
        for level in levels:
            self.check_level(level)
        self._transformed_code_level = levels

    @property
    def need_to_echo_log_to_stdout(self):
        if self._need_to_echo_log_to_stdout is not None:
            return self._need_to_echo_log_to_stdout
        return False

    @need_to_echo_log_to_stdout.setter
    def need_to_echo_log_to_stdout(self, log_to_stdout):
        assert isinstance(log_to_stdout, (bool, type(None)))
        self._need_to_echo_log_to_stdout = log_to_stdout

    @property
    def need_to_echo_code_to_stdout(self):
        if self._need_to_echo_code_to_stdout is not None:
            return self._need_to_echo_code_to_stdout
        return False

    @need_to_echo_code_to_stdout.setter
    def need_to_echo_code_to_stdout(self, code_to_stdout):
        assert isinstance(code_to_stdout, (bool, type(None)))
        self._need_to_echo_code_to_stdout = code_to_stdout

    def check_level(self, level):
        if isinstance(level, (int, type(None))):
            rv = level
        else:
            raise TypeError(f"Level is not an integer: {level}")
        return rv

    def has_code_level(self, level):
        level = self.check_level(level)
        return level in self.transformed_code_level or self.transformed_code_level == [
            -1
        ]

    def has_verbosity(self, level):
        """
        Checks whether the verbosity level set by the user is greater than or equal to the log level.
        Args:
            level(int): The level of log.
        Returns:
            True if the verbosity level set by the user is greater than or equal to the log level, otherwise False.
        """
        level = self.check_level(level)
        return self.verbosity_level >= level

    def error(self, msg, *args, **kwargs):
        if USE_LOGGING:
            self.logger.error(msg, *args, **kwargs)
            if self.need_to_echo_log_to_stdout:
                self._output_to_stdout("ERROR: " + msg, *args)

    def warn(self, msg, *args, **kwargs):
        if USE_LOGGING and self.verbosity_level != -1:
            self.logger.warning(msg, *args, **kwargs)
            if self.need_to_echo_log_to_stdout:
                self._output_to_stdout("WARNING: " + msg, *args)

    def log(self, level, msg, *args, **kwargs):
        if USE_LOGGING and self.has_verbosity(level):
            msg_with_level = f"(Level {level}) {msg}"
            self.logger.info(msg_with_level, *args, **kwargs)
            if self.need_to_echo_log_to_stdout:
                self._output_to_stdout("INFO: " + msg_with_level, *args)

    def log_transformed_code(
        self, level, ast_node, transformer_name, position="AFTER", *args, **kwargs
    ):
        if USE_LOGGING and self.has_code_level(level):
            from ivy.transpiler.utils.ast_utils import ast_to_source_code

            source_code = ast_to_source_code(ast_node)
            if level == MAX_TRANSFORMERS:
                header_msg = "{} the last level ast transformer: '{}', the transformed code:\n".format(
                    position, transformer_name
                )
            else:
                header_msg = "{} the level {} ast transformer: '{}', the transformed code:\n".format(
                    position, level, transformer_name
                )

            msg = header_msg + source_code
            self.logger.info(msg, *args, **kwargs)

            if self.need_to_echo_code_to_stdout:
                self._output_to_stdout("INFO: " + msg, *args)

    def _output_to_stdout(self, msg, *args):
        msg = self.logger_name + " " + msg
        print(msg % args)


_TRANSLATOR_LOGGER = Logger()
