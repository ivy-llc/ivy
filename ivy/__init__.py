# class placeholders

class NativeArray:
    pass


class NativeVariable:
    pass


class Array:
    pass


class Variable:
    pass


class Framework:
    pass


class Device:
    pass


class Node:
    pass


class Dtype:
    pass


# global constants
_MIN_DENOMINATOR = 1e-12
_MIN_BASE = 1e-5


# local
import ivy
from .array import Array, Variable
from .container import Container, MultiDevContainer
from ivy.array.array_mode_handler import set_array_mode, unset_array_mode, array_mode, array_mode_val
from .framework_handler import current_framework, get_framework, set_framework, unset_framework, framework_stack,\
    choose_random_framework, try_import_ivy_jax, try_import_ivy_tf, try_import_ivy_torch, try_import_ivy_mxnet,\
    try_import_ivy_numpy, clear_framework_stack
from . import framework_handler, func_wrapper
from .debugger import set_debug_mode, set_breakpoint_debug_mode, set_exception_debug_mode, unset_debug_mode,\
    debug_mode, debug_mode_val
from . import debugger
from .graph_compiler import *
from . import graph_compiler
from ivy.functional.ivy.core import *
from .functional import frontends
from .functional.ivy import nn
from ivy.functional.ivy.nn import *
from . import stateful
from .stateful import *
from . import verbosity
from .array import *
from ivy.array import ArrayWithDevice, ArrayWithGeneral, ArrayWithGradients, ArrayWithImage, ArrayWithLinalg,\
    ArrayWithLogic, ArrayWithMath, ArrayWithMeta, ArrayWithRandom, ArrayWithReductions

# data types
int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
uint8 = 'uint8'
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'
bfloat16 = 'bfloat16'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
# noinspection PyShadowingBuiltins
bool = 'bool'

all_dtypes = (int8, int16, int32, int64,
              uint8, uint16, uint32, uint64,
              bfloat16, float16, float32, float64)
valid_dtypes = all_dtypes
invalid_dtypes = ()

all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'bfloat16', 'float16', 'float32', 'float64')
valid_dtype_strs = all_dtypes
invalid_dtype_strs = ()

iinfo = None
finfo = None

backend = 'none'
