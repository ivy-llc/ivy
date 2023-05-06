from . import vision
from . import nn
from . import linalg
from . import fft
from . import signal

from .tensor.attribute import *
from .tensor.creation import *
from .tensor.linalg import *
from .tensor.logic import *
from .tensor.manipulation import *
from .tensor.math import *
from .tensor.random import *
from .tensor.search import *
from .tensor.einsum import *

from .tensor.tensor import Tensor


_frontend_array = Tensor
