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

all_dtypes = (int8, int16, int32, int64,
              uint8, uint16, uint32, uint64,
              bfloat16, float16, float32, float64)
valid_dtypes = all_dtypes

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ()


def closest_valid_dtype(type):
    if type is None:
        return ivy.default_dtype()
    return type


backend = 'tensorflow'


# local sub-modules
from . import activations
from .activations import *
from . import constants
from .constants import *
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
