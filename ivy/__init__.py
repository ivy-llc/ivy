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
from .array import Array, Variable, add_ivy_array_instance_methods
from .array.conversions import *
from .container import Container, MultiDevContainer, add_ivy_container_instance_methods
from .framework_handler import current_framework, get_framework, set_framework, unset_framework, framework_stack,\
    choose_random_framework, try_import_ivy_jax, try_import_ivy_tf, try_import_ivy_torch, try_import_ivy_mxnet,\
    try_import_ivy_numpy, clear_framework_stack
from . import framework_handler, func_wrapper
from .debugger import set_debug_mode, set_breakpoint_debug_mode, set_exception_debug_mode, unset_debug_mode,\
    debug_mode, debug_mode_val
from . import debugger
from . import functional
from .functional import *
from . import stateful
from .stateful import *
from . import verbosity

# add instance methods to Ivy Array and Container
from ivy.functional.ivy import activations, creation, data_type, device, elementwise, general, gradients, image,\
    layers, linear_algebra, losses, manipulation, norms, random, searching, set, sorting, statistical, utility
add_ivy_array_instance_methods(
    Array, [activations, creation, data_type, device, elementwise, general, gradients, image, layers, linear_algebra,
            losses, manipulation, norms, random, searching, set, sorting, statistical, utility])
add_ivy_container_instance_methods(
    Container, [activations, creation, data_type, device, elementwise, general, gradients, image, layers,
                linear_algebra, losses, manipulation, norms, random, searching, set, sorting, statistical, utility])

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
nan = float('nan')
inf = float('inf')

valid_dtypes = (int8, int16, int32, int64,
                uint8, uint16, uint32, uint64,
                bfloat16, float16, float32, float64,
                bool)
valid_numeric_dtypes = (int8, int16, int32, int64,
                        uint8, uint16, uint32, uint64,
                        bfloat16, float16, float32, float64)
valid_int_dtypes = (int8, int16, int32, int64,
                    uint8, uint16, uint32, uint64)
valid_float_dtypes = (bfloat16, float16, float32, float64)

# all
all_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'bfloat16', 'float16', 'float32', 'float64',
                  'bool')
all_numeric_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                          'uint8', 'uint16', 'uint32', 'uint64',
                          'bfloat16', 'float16', 'float32', 'float64')
all_int_dtype_strs = ('int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64')
all_float_dtype_strs = ('bfloat16', 'float16', 'float32', 'float64')

# valid
valid_dtype_strs = all_dtype_strs
valid_numeric_dtype_strs = all_numeric_dtype_strs
valid_int_dtype_strs = all_int_dtype_strs
valid_float_dtype_strs = all_float_dtype_strs

# invalid
invalid_dtype_strs = ()
invalid_numeric_dtype_strs = ()
invalid_int_dtype_strs = ()
invalid_float_dtype_strs = ()

backend = 'none'

if 'IVY_BACKEND' in os.environ:
    ivy.set_framework(os.environ['IVY_BACKEND'])
