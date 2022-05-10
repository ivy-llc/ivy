# global
import sys
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from tensorflow.python.framework.dtypes import DType

# local
import ivy

# noinspection PyUnresolvedReferences
use = ivy.framework_handler.ContextManager(sys.modules[__name__])

NativeArray = Tensor
NativeVariable = Tensor
Device = str
Dtype = DType

# data types
int8 = tf.int8
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64
uint8 = tf.uint8
uint16 = tf.uint16
uint32 = tf.uint32
uint64 = tf.uint64
bfloat16 = tf.bfloat16
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
# noinspection PyShadowingBuiltins
bool = tf.bool

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
valid_numeirc_dtypes = (
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
    return type


backend = "tensorflow"


# local sub-modules
from . import activations  # noqa
from .activations import *  # noqa
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
