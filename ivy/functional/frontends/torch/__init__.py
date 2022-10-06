# flake8: noqa
from . import tensor
from .tensor import Tensor
from . import indexing_slicing_joining_mutating_ops
from .indexing_slicing_joining_mutating_ops import *
from . import pointwise_ops
from .pointwise_ops import *
from . import non_linear_activation_functions
from .non_linear_activation_functions import *
from . import creation_ops
from .creation_ops import *
from . import comparison_ops
from .comparison_ops import *
from . import random_sampling
from .random_sampling import *
from . import reduction_ops
from .reduction_ops import *
from . import spectral_ops
from .spectral_ops import *
from . import tensor_functions
from .tensor_functions import *
from . import blas_and_lapack_ops
from .blas_and_lapack_ops import *
from . import convolution_functions
from .convolution_functions import *
from . import distance_functions
from .distance_functions import *
from . import dropout_functions
from .dropout_functions import *
from . import linear_functions
from .linear_functions import *
from . import locally_disabling_gradient_computation
from .locally_disabling_gradient_computation import *
from . import loss_functions
from .loss_functions import *
from . import miscellaneous_ops
from .miscellaneous_ops import *
from . import pooling_functions
from .pooling_functions import *
from . import sparse_functions
from .sparse_functions import *
from . import utilities
from .utilities import *
from . import vision_functions
from .vision_functions import *

# global
from torch import Tensor, device, dtype, Size


FrontendArray = Tensor
FrontendVariable = Tensor
FrontendDevice = device
FrontendDtype = dtype
FrontendShape = Size
