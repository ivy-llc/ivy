# global
import inspect
from typing import List

# local
from .logging_utils import Logger


__all__ = []


CONVERSION_OPTIONS = "_ivy_not_s2s"
logger = Logger()


class ConversionOptions:
    """
    A container for conversion flags of a function in Source-to-Source.

    Attributes:
        not_convert(bool): An attribute indicates that the function won't be converted Source-to-Source.

    NOTE: More attributes and methods can be added in this class.
    """

    def __init__(self, not_convert=False):
        self.not_convert = not_convert

    def attach(self, func):
        if inspect.ismethod(func):
            func = func.__func__

        if inspect.isfunction(func):
            setattr(func, CONVERSION_OPTIONS, self)
        else:
            logger.warn(
                "Only support @not_s2s to type(function) or type(method), but recevied {}".format(
                    type(func)
                )
            )


def not_s2s(func=None):
    """
    A Decorator to suppresses the convertion of a function.

    Args:
        func(callable): The function to decorate.

    Returns:
        callable: A function which won't be converted Source-2-Source.
    """
    if func is None:
        return not_s2s

    options = ConversionOptions(not_convert=True)
    options.attach(func)
    return func


def builtin_modules():
    """
    Return names of builtin modules.
    """
    module_names = [
        "copy",
        "collections",
        "collections.abc",
        "color_operations",
        "enum",
        "itertools",
        "operator",
        "math",
        "numbers",
        "warnings",
        "inspect",
        "logging",
        "numpy",
        "ml_dtypes",
        "scipy",
        "scipy.stats",
        "matplotlib",
        "sklearn",
        "pandas",
        "os",
        "os.path",
        "sys",
        "abc",
        "pdb",
        "re",
        "typing",
        "typing_extensions",
        "types",
        "dataclasses",
        "builtins",
        "functools",
        "packaging",
        "packaging.version",
        "shutil",
        "psutil",
        "tempfile",
        "zipfile",
        "json",
        "gc",
        "contextlib",
        "pathlib",
        "uuid",
        "urllib",
        "concurrent",
        "traceback",
        "requests",
        "io",
        "threading",
        "ast",
        "gast",
        "six",
        "asyncio",
        "base64",
        "datetime",
        "email",
        "importlib",
        "weakref",
        "tqdm",
        "platform",
        "pprint",
        "selectors",
        "socket",
        "subprocess",
        "aiohttp",
        "filelock",
        "filelock",
        "jinja2",
        "markupsafe",
        "PIL",
        "pyarrow",
        "yaml",
        "posixpath",
        "genericpath",
    ]

    return module_names


BUILTIN_LIKELY_MODULE_NAMES = builtin_modules()


def add_ignore_module(module_names: List[str]):
    """
    Adds module names that should be ignored during translation.
    """
    global BUILTIN_LIKELY_MODULE_NAMES
    for module_name in module_names:
        if module_name not in BUILTIN_LIKELY_MODULE_NAMES:
            BUILTIN_LIKELY_MODULE_NAMES.append(module_name)


def is_builtin_function(func):
    """
    Checks whether the function is from a builtin module by comparing module names.
    """
    if hasattr(func, "__module__"):
        module_name = func.__module__.split(".")[0]
        if module_name in BUILTIN_LIKELY_MODULE_NAMES:
            logger.log(
                2,
                f"Whitelist: {func} is part of built-in module '{module_name}' and does not have to be transformed.",
            )
            return True
    return False
