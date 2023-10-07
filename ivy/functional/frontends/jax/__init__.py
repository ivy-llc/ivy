# global
import sys

from . import _src
from . import array
from . import config
from . import general_functions
from . import lax
from . import nn
from . import numpy
from . import random
from ._src import tree_util
from .array import *
from .general_functions import *
from ivy.functional.frontends import set_frontend_to_specific_version

# local


_frontend_array = numpy.array

# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

__version__ = set_frontend_to_specific_version(module)
