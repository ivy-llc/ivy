# flake8: noqa
from . import devicearray
from .devicearray import DeviceArray
from .lax import operators
from ivy.functional.frontends.jax.lax.operators import *
from .lax import control_flow_operators
from ivy.functional.frontends.jax.lax.control_flow_operators import *
from .lax import custom_gradient_operators
from ivy.functional.frontends.jax.lax.custom_gradient_operators import *
from .lax import parallel_operators
from ivy.functional.frontends.jax.lax.parallel_operators import *
from .lax import linalg
from ivy.functional.frontends.jax.lax.linalg import *
from .nn import non_linear_activations
from ivy.functional.frontends.jax.nn.non_linear_activations import *
from .numpy import name_space_functions
from ivy.functional.frontends.jax.numpy.name_space_functions import *
from .numpy import fft
from ivy.functional.frontends.jax.numpy.fft import *
from .numpy import linalg
from ivy.functional.frontends.jax.numpy.linalg import *

# global
from jax.numpy import dtype
from jax.interpreters.xla import _DeviceArray
from jaxlib.xla_extension import DeviceArray, Device, Buffer


FrontendArray = (
    _DeviceArray,
    DeviceArray,
    Buffer,
)
FrontendVariable = _DeviceArray
FrontendDevice = Device
FrontendDtype = dtype
FrontendShape = tuple
