# global
import sys
from jax.config import config
config.update("jax_enable_x64", True)
import jaxlib
import jax as _jax
import jax.numpy as jnp
from typing import Union
# noinspection PyPackageRequirements
from jaxlib.xla_extension import Buffer

# make ivy.Container compatible with jax pytree traversal
from jax.tree_util import register_pytree_node
from jax.tree_util import tree_flatten, tree_unflatten

# local
import ivy

register_pytree_node(
    ivy.Container,
    lambda c: tree_flatten(c.to_dict()),
    lambda a, c: ivy.Container(tree_unflatten(a, c))
)

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

# noinspection PyUnresolvedReferences
JaxArray = Union[_jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer]
# noinspection PyUnresolvedReferences,PyProtectedMember
NativeArray = (_jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer)
# noinspection PyUnresolvedReferences,PyProtectedMember
NativeVariable = _jax.interpreters.xla._DeviceArray
# noinspection PyUnresolvedReferences
Device = jaxlib.xla_extension.Device
Dtype = jnp.dtype

# data types
int8 = jnp.dtype('int8')
int16 = jnp.dtype('int16')
int32 = jnp.dtype('int32')
int64 = jnp.dtype('int64')
uint8 = jnp.dtype('uint8')
uint16 = jnp.dtype('uint16')
uint32 = jnp.dtype('uint32')
uint64 = jnp.dtype('uint64')
bfloat16 = jnp.dtype('bfloat16')
float16 = jnp.dtype('float16')
float32 = jnp.dtype('float32')
float64 = jnp.dtype('float64')
# noinspection PyShadowingBuiltins
bool = jnp.dtype('bool')

all_dtypes = (int8, int16, int32,
              uint8, uint16, uint32, uint64,
              bfloat16, float16, float32)
valid_dtypes = all_dtypes
invalid_dtypes = ()

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = dtype_to_str(type)
    if type_str in invalid_dtype_strs:
        return {'int64': int32,
                'uint64': uint32,
                'float64': float32}[type_str]
    return type


backend = 'jax'


# local sub-modules
from . import activations
from .activations import *
from . import constants
from .constants import *
from . import converters
from .converters import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import elementwise
from .elementwise import *
from . import layers
from .layers import *
from . import linear_algebra
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import searching
from .searching import *
from . import set
from .set import *
from . import sorting
from .sorting import *
from . import statistical
from .statistical import *
from . import utility
from .utility import *

from . import old
from .old import *
