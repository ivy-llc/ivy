# global
import sys

# local
from ivy.functional.frontends import set_frontend_to_specific_version
from . import config
from . import devicearray
from .devicearray import DeviceArray
from . import general_functions
from .general_functions import *
from . import lax
from . import nn
from . import numpy
from . import random
from . import _src
from ._src import tree_util


_frontend_array = numpy.array

# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

set_frontend_to_specific_version(module)
