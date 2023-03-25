from .test_array_api import array_api_tests
from .test_array_api.array_api_tests import *
from .test_array_api import *
from . import array_api_methods_to_test
from . import test_array_api


try:
    from jax.config import config

    config.update("jax_enable_x64", True)
except (ImportError, RuntimeError):
    pass
