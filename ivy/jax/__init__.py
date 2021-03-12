import sys
import ivy
import jax
# noinspection PyPackageRequirements
import jaxlib
import jax.numpy as jnp

from .core import *
from . import nn
from .nn import *

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

# noinspection PyUnresolvedReferences,PyProtectedMember
Array = jax.interpreters.xla._DeviceArray
# noinspection PyUnresolvedReferences,PyProtectedMember
Variable = jax.interpreters.xla._DeviceArray
# noinspection PyUnresolvedReferences
Device = jaxlib.xla_extension.Device
Dtype = jnp.dtype

backend = 'jax'
