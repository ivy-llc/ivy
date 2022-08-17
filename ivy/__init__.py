# global
import builtins
import warnings

warnings.filterwarnings("ignore", module="^(?!.*ivy).*$")

# class placeholders


class Container:
    pass


class NativeArray:
    pass


class NativeVariable:
    pass


class Array:
    pass


class Variable:
    pass


class FrameworkStr(str):
    def __new__(cls, fw_str):
        assert fw_str in ["jax", "tensorflow", "torch", "mxnet", "numpy"]
        return str.__new__(cls, fw_str)


class Framework:
    pass


class NativeDevice:
    pass


class NativeDtype:
    pass


class NativeShape:
    pass


class Device(str):
    def __new__(cls, dev_str):
        if dev_str != "":
            assert dev_str[0:3] in ["gpu", "tpu", "cpu"]
            if dev_str != "cpu":
                assert dev_str[3] == ":"
                assert dev_str[4:].isnumeric()
        return str.__new__(cls, dev_str)


class Dtype(str):
    def __new__(cls, dtype_str):
        assert "int" in dtype_str or "float" in dtype_str or "bool" in dtype_str
        return str.__new__(cls, dtype_str)


class Shape(tuple):
    def __new__(cls, shape_tup):
        valid_types = (int, list, tuple)
        if len(backend_stack) != 0:
            valid_types += (ivy.NativeShape,)
        assert isinstance(shape_tup, valid_types)
        if isinstance(shape_tup, int):
            shape_tup = (shape_tup,)
        elif isinstance(shape_tup, list):
            shape_tup = tuple(shape_tup)
        assert builtins.all(
            [isinstance(v, int) or ivy.is_int_dtype(v.dtype) for v in shape_tup]
        )
        if ivy.shape_array_mode():
            return ivy.array(shape_tup)
        return tuple.__new__(cls, shape_tup)


class IntDtype(Dtype):
    def __new__(cls, dtype_str):
        assert "int" in dtype_str
        return str.__new__(cls, dtype_str)


class FloatDtype(Dtype):
    def __new__(cls, dtype_str):
        assert "float" in dtype_str
        return str.__new__(cls, dtype_str)


class UintDtype(IntDtype):
    def __new__(cls, dtype_str):
        assert "uint" in dtype_str
        return str.__new__(cls, dtype_str)


class Node(str):
    # ToDo: add formatting checks once multi-node is supported
    pass


array_significant_figures_stack = list()
array_decimal_values_stack = list()
warning_level_stack = list()
warn_to_regex = {"all": "!.*", "ivy_only": "^(?!.*ivy).*$", "none": ".*"}


# global constants
_MIN_DENOMINATOR = 1e-12
_MIN_BASE = 1e-5


# local
import threading
from .array import Array, Variable, add_ivy_array_instance_methods
from .array.conversions import *
from .array import conversions as arr_conversions
from .container import conversions as cont_conversions
from .container import (
    ContainerBase,
    Container,
    add_ivy_container_instance_methods,
)
from .backend_handler import (
    current_backend,
    get_backend,
    set_backend,
    unset_backend,
    backend_stack,
    choose_random_backend,
    try_import_ivy_jax,
    try_import_ivy_tf,
    try_import_ivy_torch,
    try_import_ivy_mxnet,
    try_import_ivy_numpy,
    clear_backend_stack,
)
from . import backend_handler, func_wrapper
from . import functional
from .functional import *
from . import stateful
from .stateful import *
from . import verbosity
from .inspection import fn_array_spec, add_array_specs

add_array_specs()

# add instance methods to Ivy Array and Container
from ivy.functional.ivy import (
    activations,
    creation,
    data_type,
    device,
    elementwise,
    general,
    gradients,
    layers,
    linear_algebra,
    losses,
    manipulation,
    norms,
    random,
    searching,
    set,
    sorting,
    statistical,
    utility,
)

add_ivy_array_instance_methods(
    Array,
    [
        activations,
        arr_conversions,
        creation,
        data_type,
        device,
        elementwise,
        general,
        gradients,
        layers,
        linear_algebra,
        losses,
        manipulation,
        norms,
        random,
        searching,
        set,
        sorting,
        statistical,
        utility,
    ],
)

add_ivy_container_instance_methods(
    Container,
    [
        activations,
        cont_conversions,
        creation,
        data_type,
        device,
        elementwise,
        general,
        gradients,
        layers,
        linear_algebra,
        losses,
        manipulation,
        norms,
        random,
        searching,
        set,
        sorting,
        statistical,
        utility,
    ],
)


add_ivy_container_instance_methods(
    Container,
    [
        activations,
        cont_conversions,
        creation,
        data_type,
        device,
        elementwise,
        general,
        gradients,
        layers,
        linear_algebra,
        losses,
        manipulation,
        norms,
        random,
        searching,
        set,
        sorting,
        statistical,
        utility,
    ],
    static=True,
)

# data types
int8 = IntDtype("int8")
int16 = IntDtype("int16")
int32 = IntDtype("int32")
int64 = IntDtype("int64")
uint8 = UintDtype("uint8")
uint16 = UintDtype("uint16")
uint32 = UintDtype("uint32")
uint64 = UintDtype("uint64")
bfloat16 = FloatDtype("bfloat16")
float16 = FloatDtype("float16")
float32 = FloatDtype("float32")
float64 = FloatDtype("float64")
# noinspection PyShadowingBuiltins
bool = Dtype("bool")

# native data types
native_int8 = IntDtype("int8")
native_int16 = IntDtype("int16")
native_int32 = IntDtype("int32")
native_int64 = IntDtype("int64")
native_uint8 = UintDtype("uint8")
native_uint16 = UintDtype("uint16")
native_uint32 = UintDtype("uint32")
native_uint64 = UintDtype("uint64")
native_bfloat16 = FloatDtype("bfloat16")
native_float16 = FloatDtype("float16")
native_float32 = FloatDtype("float32")
native_float64 = FloatDtype("float64")
native_bool = Dtype("bool")

# all
all_dtypes = (
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
all_numeric_dtypes = (
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
all_int_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
all_float_dtypes = (
    bfloat16,
    float16,
    float32,
    float64,
)
all_uint_dtypes = (
    uint8,
    uint16,
    uint32,
    uint64,
)

# valid data types
valid_dtypes = all_dtypes
valid_numeric_dtypes = all_numeric_dtypes
valid_int_dtypes = all_int_dtypes
valid_float_dtypes = all_float_dtypes
valid_uint_dtypes = all_uint_dtypes

# invalid data types
invalid_dtypes = ()
invalid_numeric_dtypes = ()
invalid_int_dtypes = ()
invalid_float_dtypes = ()
invalid_uint_dtypes = ()

# data type promotion
array_api_promotion_table = {
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, uint8): uint64,
    (uint64, uint16): uint64,
    (uint64, uint32): uint64,
    (uint64, uint64): uint64,
    (int8, uint8): int16,
    (int8, uint16): int32,
    (int8, uint32): int64,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    (uint8, int8): int16,
    (uint16, int8): int32,
    (uint32, int8): int64,
    (uint8, int16): int16,
    (uint16, int16): int32,
    (uint32, int16): int64,
    (uint8, int32): int32,
    (uint16, int32): int32,
    (uint32, int32): int64,
    (uint8, int64): int64,
    (uint16, int64): int64,
    (uint32, int64): int64,
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, float16): float32,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float16): float64,
    (float64, float32): float64,
    (float64, float64): float64,
    (bool, bool): bool,
}
locks = {"backend_setter": threading.Lock()}
extra_promotion_table = {
    (int8, float16): float16,
    (float16, int8): float16,
    (int8, float32): float32,
    (float32, int8): float32,
    (int8, float64): float64,
    (float64, int8): float64,
    (int16, float16): float16,
    (float16, int16): float16,
    (int16, float32): float32,
    (float32, int16): float32,
    (int16, float64): float64,
    (float64, int16): float64,
    (int32, float16): float16,
    (float16, int32): float16,
    (int32, float32): float32,
    (float32, int32): float32,
    (int32, float64): float64,
    (float64, int32): float64,
    (int64, float16): float16,
    (float16, int64): float16,
    (int64, float32): float32,
    (float32, int64): float32,
    (int64, float64): float64,
    (float64, int64): float64,
    (uint8, float16): float16,
    (float16, uint8): float16,
    (uint8, float32): float32,
    (float32, uint8): float32,
    (uint8, float64): float64,
    (float64, uint8): float64,
    (uint16, float16): float16,
    (float16, uint16): float16,
    (uint16, float32): float32,
    (float32, uint16): float32,
    (uint16, float64): float64,
    (float64, uint16): float64,
    (uint32, float16): float16,
    (float16, uint32): float16,
    (uint32, float32): float32,
    (float32, uint32): float32,
    (uint32, float64): float64,
    (float64, uint32): float64,
    (uint64, float16): float16,
    (float16, uint64): float16,
    (uint64, float32): float32,
    (float32, uint64): float32,
    (uint64, float64): float64,
    (float64, uint64): float64,
    (bfloat16, bfloat16): bfloat16,
}

promotion_table = {**array_api_promotion_table, **extra_promotion_table}


backend = "none"

native_inplace_support = None

supports_gradients = None

if "IVY_BACKEND" in os.environ:
    ivy.set_backend(os.environ["IVY_BACKEND"])

# Array Significant Figures #


def _assert_array_significant_figures_formatting(sig_figs):
    assert isinstance(sig_figs, int)
    assert sig_figs > 0


def _sf(x, sig_fig=3):
    if isinstance(x, np.bool_):
        return x
    f = float(
        np.format_float_positional(
            x, precision=sig_fig, unique=False, fractional=False, trim="k"
        )
    )
    if "uint" in type(x).__name__:
        f = np.uint(f)
    elif "int" in type(x).__name__:
        f = int(f)
    x = f
    return x


vec_sig_fig = np.vectorize(_sf)
vec_sig_fig.__name__ = "vec_sig_fig"


def array_significant_figures(sig_figs=None):
    """Summary.

    Parameters
    ----------
    sig_figs
        optional int, number of significant figures to be shown when printing

    Returns
    -------
    ret

    """
    if ivy.exists(sig_figs):
        _assert_array_significant_figures_formatting(sig_figs)
        return sig_figs
    global array_significant_figures_stack
    if not array_significant_figures_stack:
        ret = 3
    else:
        ret = array_significant_figures_stack[-1]
    return ret


def set_array_significant_figures(sig_figs):
    """Summary.

    Parameters
    ----------
    sig_figs
        optional int, number of significant figures to be shown when printing

    """
    _assert_array_significant_figures_formatting(sig_figs)
    global array_significant_figures_stack
    array_significant_figures_stack.append(sig_figs)


def unset_array_significant_figures():
    """"""
    global array_significant_figures_stack
    if array_significant_figures_stack:
        array_significant_figures_stack.pop(-1)


# Decimal Values #


def _assert_array_decimal_values_formatting(dec_vals):
    assert isinstance(dec_vals, int)
    assert dec_vals >= 0


def array_decimal_values(dec_vals=None):
    """Summary.

    Parameters
    ----------
    dec_vals
        optional int, number of decimal values to be shown when printing

    Returns
    -------
    ret

    """
    if ivy.exists(dec_vals):
        _assert_array_decimal_values_formatting(dec_vals)
        return dec_vals
    global array_decimal_values_stack
    if not array_decimal_values_stack:
        ret = None
    else:
        ret = array_decimal_values_stack[-1]
    return ret


def set_array_decimal_values(dec_vals):
    """Summary.

    Parameters
    ----------
    dec_vals
        optional int, number of significant figures to be shown when printing

    """
    _assert_array_decimal_values_formatting(dec_vals)
    global array_decimal_values_stack
    array_decimal_values_stack.append(dec_vals)


def unset_array_decimal_values():
    """"""
    global array_decimal_values_stack
    if array_decimal_values_stack:
        array_decimal_values_stack.pop(-1)


def warning_level():
    """Summary.

    Returns
    -------
    ret
        current warning level, default is "ivy_only"
    """
    global warning_level_stack
    if not warning_level_stack:
        ret = "ivy_only"
    else:
        ret = warning_level_stack[-1]
    return ret


def set_warning_level(warn_level):
    """Summary.

    Parameters
    ----------
    warn_level
        string for the warning level to be set, one of "none", "ivy_only", "all"

    """
    global warning_level_stack
    warning_level_stack.append(warn_level)


def unset_warning_level():
    """"""
    global warning_level_stack
    if warning_level_stack:
        warning_level_stack.pop(-1)


def warn(warning_message, stacklevel=0):
    warn_level = warning_level()
    warnings.filterwarnings("ignore", module=warn_to_regex[warn_level])
    warnings.warn(warning_message, stacklevel=stacklevel)
