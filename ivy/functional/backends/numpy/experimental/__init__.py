# global
import numpy as np

backend_version = {"version": np.__version__}

# local sub-modules
# flake8: noqa
from .activations import *
from .compilation import *
from .creation import *
from .data_type import *
from .device import *
from .elementwise import *
from .general import *
from .gradients import *
from .layers import *
from .linear_algebra import *
from .manipulation import *
from .random import *
from .searching import *
from .set import *
from .sorting import *
from .sparse_array import *
from .statistical import *
from .utility import *
