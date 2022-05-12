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
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

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
Device = jaxlib.xla_extension.Device
Dtype = jnp.dtype

# data types
int8 = jnp.dtype("int8")
int16 = jnp.dtype("int16")
int32 = jnp.dtype("int32")
int64 = jnp.dtype("int64")
uint8 = jnp.dtype("uint8")
uint16 = jnp.dtype("uint16")
uint32 = jnp.dtype("uint32")
uint64 = jnp.dtype("uint64")
bfloat16 = jnp.dtype("bfloat16")
float16 = jnp.dtype("float16")
float32 = jnp.dtype("float32")
float64 = jnp.dtype("float64")
# noinspection PyShadowingBuiltins
bool = jnp.dtype("bool")

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

# valid
valid_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "bool",
)
valid_numeric_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
)
valid_int_dtype_strs = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
valid_float_dtype_strs = ("bfloat16", "float16", "float32", "float64")

# invalid
invalid_dtype_strs = ()
invalid_numeric_dtype_strs = ()
invalid_int_dtype_strs = ()
invalid_float_dtype_strs = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    type_str = dtype_to_str(type)  # noqa
    if type_str in invalid_dtype_strs:
        return {"int64": int32, "uint64": uint32, "float64": float32}[type_str]
    return type


backend = "jax"

# local sub-modules
from . import activations  # noqa
from .activations import *  # noqa
from . import converters  # noqa
from .converters import *  # noqa
from . import creation  # noqa
from .creation import *  # noqa
from . import data_type  # noqa
from .data_type import *  # noqa
from . import device  # noqa
from .device import *  # noqa
from . import elementwise  # noqa
from .elementwise import *  # noqa
from . import general  # noqa
from .general import *  # noqa
from . import gradients  # noqa
from .gradients import *  # noqa
from . import image  # noqa
from .image import *  # noqa
from . import layers  # noqa
from .layers import *  # noqa
from . import linear_algebra as linalg  # noqa
from .linear_algebra import *  # noqa
from . import manipulation  # noqa
from .manipulation import *  # noqa
from . import random  # noqa
from .random import *  # noqa
from . import searching  # noqa
from .searching import *  # noqa
from . import set  # noqa
from .set import *  # noqa
from . import sorting  # noqa
from .sorting import *  # noqa
from . import statistical  # noqa
from .statistical import *  # noqa
from . import utility  # noqa
from .utility import *  # noqa
