# global
import sys
from jax.config import config
import jaxlib
import jax as jax
import jax.numpy as jnp
from typing import Union

# noinspection PyPackageRequirements
from jaxlib.xla_extension import Buffer

# make ivy.Container compatible with jax pytree traversal
from jax.tree_util import register_pytree_node
from jax.tree_util import tree_flatten, tree_unflatten

# local
import ivy

config.update("jax_enable_x64", True)

register_pytree_node(
    ivy.Container,
    lambda c: tree_flatten(c.to_dict()),
    lambda a, c: ivy.Container(tree_unflatten(a, c)),
)

# noinspection PyUnresolvedReferences
use = ivy.backend_handler.ContextManager(sys.modules[__name__])

# noinspection PyUnresolvedReferences
JaxArray = Union[
    jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer
]
# noinspection PyUnresolvedReferences,PyProtectedMember
NativeArray = (
    jax.interpreters.xla._DeviceArray,
    jaxlib.xla_extension.DeviceArray,
    Buffer,
)
# noinspection PyUnresolvedReferences,PyProtectedMember
NativeVariable = jax.interpreters.xla._DeviceArray
# noinspection PyUnresolvedReferences
NativeDevice = jaxlib.xla_extension.Device
NativeDtype = jnp.dtype

# data types
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.IntDtype("uint8")
uint16 = ivy.IntDtype("uint16")
uint32 = ivy.IntDtype("uint32")
uint64 = ivy.IntDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
# noinspection PyShadowingBuiltins
bool = "bool"
nan = float("nan")
inf = float("inf")

# native data types
native_int8 = jnp.dtype("int8")
native_int16 = jnp.dtype("int16")
native_int32 = jnp.dtype("int32")
native_int64 = jnp.dtype("int64")
native_uint8 = jnp.dtype("uint8")
native_uint16 = jnp.dtype("uint16")
native_uint32 = jnp.dtype("uint32")
native_uint64 = jnp.dtype("uint64")
native_bfloat16 = jnp.dtype("bfloat16")
native_float16 = jnp.dtype("float16")
native_float32 = jnp.dtype("float32")
native_float64 = jnp.dtype("float64")
# noinspection PyShadowingBuiltins
native_bool = jnp.dtype("bool")

valid_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bfloat16,
    float16,
    float32,
    float64,
    bool,
)
valid_numeric_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bfloat16,
    float16,
    float32,
    float64,
)
valid_int_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
valid_float_dtypes = (bfloat16, float16, float32, float64)

# invalid
invalid_dtypes = ()
invalid_numeric_dtypes = ()
invalid_int_dtypes = ()
invalid_float_dtypes = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = as_ivy_dtype(type)  # noqa
    if type_str in invalid_dtypes:
        return {"int64": int32, "uint64": uint32, "float64": float32}[type_str]
    return type


backend = "jax"

# local sub-modules
from . import activations
from .activations import *
from . import compilation
from .compilation import *
from . import converters
from .converters import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import device
from .device import *
from . import elementwise
from .elementwise import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import image
from .image import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import random
from .random import *
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
