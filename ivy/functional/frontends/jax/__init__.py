# global
import sys

# local
from ivy.functional.frontends import set_frontend_to_specific_version

from . import _src, array, config, general_functions, lax, nn, numpy, random
from ._src import tree_util
from .array import *
from .general_functions import *

_frontend_array = numpy.array

# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

__version__ = set_frontend_to_specific_version(module)
