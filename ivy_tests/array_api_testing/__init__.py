from . import array_api_methods_to_test
from . import test_array_api
from .test_array_api import *

try:
    from jax.config import config

    config.update("jax_enable_x64", True)
except (ImportError, RuntimeError):
    pass
