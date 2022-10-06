# flake8: noqa
from . import tensor
from .tensor import Tensor
from . import activations
from .activations import *
from . import layers
from .layers import *
from . import linalg
from .linalg import *
from . import math
from .math import *
from . import metrics
from .metrics import *
from . import nest
from .nest import *
from . import nn
from .nn import *
from . import quantization
from .quantization import *
from . import random
from .random import *
from . import raw_ops
from .raw_ops import *
from . import regularizers
from .regularizers import *
from . import sets
from .sets import *
from . import signal
from .signal import *
from . import sparse
from .sparse import *

# global
from tensorflow.python.types.core import Tensor
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.tensor_shape import TensorShape


FrontendArray = Tensor
FrontendVariable = Tensor
FrontendDevice = str
FrontendDtype = DType
FrontendShape = TensorShape
