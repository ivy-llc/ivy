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

set_frontend_to_specific_version(sys.modules[__name__])
