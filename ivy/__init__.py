# global
from types import SimpleNamespace
import warnings
from ivy._version import __version__ as __version__
import builtins
import numpy as np

try:
    import torch
except ImportError:
    torch = SimpleNamespace()
    torch.Size = SimpleNamespace()
    torch.Tensor = SimpleNamespace()

try:
    import tensorflow as tf
except ImportError:
    tf = SimpleNamespace()
    tf.TensorShape = SimpleNamespace()
    tf.Tensor = SimpleNamespace()

try:
    import jax
    import jaxlib
except ImportError:
    jax = SimpleNamespace()
    jax.interpreters.xla._DeviceArray = SimpleNamespace()
    jaxlib.xla_extension.DeviceArray = SimpleNamespace()
    jax.Buffer = SimpleNamespace()

warnings.filterwarnings("ignore", module="^(?!.*ivy).*$")


# class placeholders


class FrameworkStr(str):
    def __new__(cls, fw_str):
        ivy.assertions.check_elem_in_list(
            fw_str, ["jax", "tensorflow", "torch", "numpy"]
        )
        return str.__new__(cls, fw_str)


class Framework:
    pass


class NativeArray:
    pass


class NativeVariable:
    pass


class NativeDevice:
    pass


class NativeDtype:
    pass


class NativeShape:
    pass


class Container:
    pass


class Array:
    pass


class Device(str):
    def __new__(cls, dev_str):
        if dev_str != "":
            ivy.assertions.check_elem_in_list(dev_str[0:3], ["gpu", "tpu", "cpu"])
            if dev_str != "cpu":
                # ivy.assertions.check_equal(dev_str[3], ":")
                ivy.assertions.check_true(
                    dev_str[4:].isnumeric(),
                    message="{} must be numeric".format(dev_str[4:]),
                )
        return str.__new__(cls, dev_str)


class Dtype(str):
    def __new__(cls, dtype_str):
        if dtype_str is builtins.int:
            dtype_str = default_int_dtype()
        if dtype_str is builtins.float:
            dtype_str = default_float_dtype()
        if dtype_str is builtins.complex:
            dtype_str = default_complex_dtype()
        if dtype_str is builtins.bool:
            dtype_str = "bool"
        if not isinstance(dtype_str, str):
            raise ivy.exceptions.IvyException("dtype must be type str")
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.exceptions.IvyException(f"{dtype_str} is not supported by ivy")
        return str.__new__(cls, dtype_str)

    def __ge__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self == ivy.promote_types(self, other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self >= other and self != other

    def __lt__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self != ivy.promote_types(self, other)

    def __le__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self < other or self == other

    @property
    def is_bool_dtype(self):
        return is_bool_dtype(self)

    @property
    def is_int_dtype(self):
        return is_int_dtype(self)

    @property
    def is_float_dtype(self):
        return is_float_dtype(self)

    @property
    def is_uint_dtype(self):
        return is_uint_dtype(self)

    @property
    def is_complex_dtype(self):
        return is_complex_dtype(self)

    @property
    def dtype_bits(self):
        return dtype_bits(self)

    @property
    def as_native_dtype(self):
        return as_native_dtype(self)

    @property
    def info(self):
        if self.is_int_dtype or self.is_uint_dtype:
            return iinfo(self)
        elif self.is_float_dtype:
            return finfo(self)
        else:
            raise ivy.exceptions.IvyError(f"{self} is not supported by info")

    def can_cast(self, to):
        return can_cast(self, to)


class Shape(tuple):
    def __new__(cls, shape_tup):
        valid_types = (int, list, tuple, ivy.Array)
        if len(backend_stack) != 0:
            valid_types += (ivy.NativeShape, ivy.NativeArray)
        else:
            valid_types += (
                tf.TensorShape,
                torch.Size,
                jax.interpreters.xla._DeviceArray,
                jaxlib.xla_extension.DeviceArray,
                jax.Buffer,
                np.ndarray,
                tf.Tensor,
            )
        ivy.assertions.check_isinstance(shape_tup, valid_types)
        if isinstance(shape_tup, int):
            shape_tup = (shape_tup,)
        elif isinstance(shape_tup, list):
            shape_tup = tuple(shape_tup)
        ivy.assertions.check_all(
            [isinstance(v, int) or ivy.is_int_dtype(v.dtype) for v in shape_tup],
            "shape must take integers only",
        )
        if ivy.shape_array_mode():
            return ivy.array(shape_tup)
        return tuple.__new__(cls, shape_tup)


class IntDtype(Dtype):
    def __new__(cls, dtype_str):
        if dtype_str is builtins.int:
            dtype_str = default_int_dtype()
        if not isinstance(dtype_str, str):
            raise ivy.exceptions.IvyException("dtype_str must be type str")
        if "int" not in dtype_str:
            raise ivy.exceptions.IvyException(
                "dtype must be string and starts with int"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.exceptions.IvyException(f"{dtype_str} is not supported by ivy")
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return iinfo(self)


class FloatDtype(Dtype):
    def __new__(cls, dtype_str):
        if dtype_str is builtins.float:
            dtype_str = default_float_dtype()
        if not isinstance(dtype_str, str):
            raise ivy.exceptions.IvyException("dtype_str must be type str")
        if "float" not in dtype_str:
            raise ivy.exceptions.IvyException(
                "dtype must be string and starts with float"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.exceptions.IvyException(f"{dtype_str} is not supported by ivy")
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return finfo(self)


class UintDtype(IntDtype):
    def __new__(cls, dtype_str):
        if not isinstance(dtype_str, str):
            raise ivy.exceptions.IvyException("dtype_str must be type str")
        if "uint" not in dtype_str:
            raise ivy.exceptions.IvyException(
                "dtype must be string and starts with uint"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.exceptions.IvyException(f"{dtype_str} is not supported by ivy")
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return iinfo(self)


class ComplexDtype(Dtype):
    def __new__(cls, dtype_str):
        if not isinstance(dtype_str, str):
            raise ivy.exceptions.IvyException("dtype_str must be type str")
        if "complex" not in dtype_str:
            raise ivy.exceptions.IvyException(
                "dtype must be string and starts with complex"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.exceptions.IvyException(f"{dtype_str} is not supported by ivy")
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return finfo(self)


class Node(str):
    # ToDo: add formatting checks once multi-node is supported
    pass


array_significant_figures_stack = list()
array_decimal_values_stack = list()
warning_level_stack = list()
nan_policy_stack = list()
warn_to_regex = {"all": "!.*", "ivy_only": "^(?!.*ivy).*$", "none": ".*"}


# global constants
_MIN_DENOMINATOR = 1e-12
_MIN_BASE = 1e-5


# local
import threading


# devices
# ToDo: add gpu and tpu for valid devices when we test for them
all_devices = ("cpu", "gpu", "tpu")

valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")

# data types as string (to be used by Dtype classes)
# any changes here should also be reflected in the data type initialisation underneath
_all_ivy_dtypes_str = (
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
    "complex64",
    "complex128",
    "bool",
)

# data types
# any changes here should also be reflected in the data type string tuple above
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
double = float64
complex64 = ComplexDtype("complex64")
complex128 = ComplexDtype("complex128")
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
native_double = native_float64
native_complex64 = ComplexDtype("complex64")
native_complex128 = ComplexDtype("complex128")
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
    complex64,
    complex128,
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
all_complex_dtypes = (
    complex64,
    complex128,
)

# valid data types
valid_dtypes = all_dtypes
valid_numeric_dtypes = all_numeric_dtypes
valid_int_dtypes = all_int_dtypes
valid_float_dtypes = all_float_dtypes
valid_uint_dtypes = all_uint_dtypes
valid_complex_dtypes = all_complex_dtypes

# invalid data types
invalid_dtypes = ()
invalid_numeric_dtypes = ()
invalid_int_dtypes = ()
invalid_float_dtypes = ()
invalid_uint_dtypes = ()
invalid_complex_dtypes = ()

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
    (bool, uint16): uint16,
    (bool, int32): int32,
    (bool, float16): float16,
    (bool, uint64): uint64,
    (bool, float64): float64,
    (bool, int8): int8,
    (bool, int64): int64,
    (bool, int16): int16,
    (bool, bfloat16): bfloat16,
    (bool, uint32): uint32,
    (bool, uint8): uint8,
    (bool, float32): float32,
    (bool, complex64): complex64,
    (bool, complex128): complex128,
    (uint16, bool): uint16,
    (int32, bool): int32,
    (float16, bool): float16,
    (uint64, bool): uint64,
    (float64, bool): float64,
    (int8, bool): int8,
    (int64, bool): int64,
    (int16, bool): int16,
    (bfloat16, bool): bfloat16,
    (uint32, bool): uint32,
    (uint8, bool): uint8,
    (float32, bool): float32,
    (complex64, bool): complex64,
    (complex128, bool): complex128,
    (uint64, int8): float64,
    (int8, uint64): float64,
    (uint64, int16): float64,
    (int16, uint64): float64,
    (uint64, int32): float64,
    (int32, uint64): float64,
    (uint64, int64): float64,
    (int64, uint64): float64,
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
    (bfloat16, uint8): bfloat16,
    (uint8, bfloat16): bfloat16,
    (bfloat16, uint16): bfloat16,
    (uint16, bfloat16): bfloat16,
    (bfloat16, uint32): bfloat16,
    (uint32, bfloat16): bfloat16,
    (bfloat16, uint64): bfloat16,
    (uint64, bfloat16): bfloat16,
    (bfloat16, int8): bfloat16,
    (int8, bfloat16): bfloat16,
    (bfloat16, int16): bfloat16,
    (int16, bfloat16): bfloat16,
    (bfloat16, int32): bfloat16,
    (int32, bfloat16): bfloat16,
    (bfloat16, int64): bfloat16,
    (int64, bfloat16): bfloat16,
    (bfloat16, float16): float32,
    (float16, bfloat16): float32,
    (bfloat16, float32): float32,
    (float32, bfloat16): float32,
    (bfloat16, float64): float64,
    (float64, bfloat16): float64,
    (complex64, int8): complex64,
    (int8, complex64): complex64,
    (complex64, int16): complex64,
    (int16, complex64): complex64,
    (complex64, int32): complex64,
    (int32, complex64): complex64,
    (complex64, int64): complex64,
    (int64, complex64): complex64,
    (complex64, uint8): complex64,
    (uint8, complex64): complex64,
    (complex64, uint16): complex64,
    (uint16, complex64): complex64,
    (complex64, uint32): complex64,
    (uint32, complex64): complex64,
    (complex64, uint64): complex64,
    (uint64, complex64): complex64,
    (complex64, float16): complex64,
    (float16, complex64): complex64,
    (complex64, float32): complex64,
    (float32, complex64): complex64,
    (complex64, float64): complex128,
    (float64, complex64): complex128,
    (complex64, bfloat16): complex64,
    (bfloat16, complex64): complex64,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, int8): complex128,
    (int8, complex128): complex128,
    (complex128, int16): complex128,
    (int16, complex128): complex128,
    (complex128, int32): complex128,
    (int32, complex128): complex128,
    (complex128, int64): complex128,
    (int64, complex128): complex128,
    (complex128, uint8): complex128,
    (uint8, complex128): complex128,
    (complex128, uint16): complex128,
    (uint16, complex128): complex128,
    (complex128, uint32): complex128,
    (uint32, complex128): complex128,
    (complex128, uint64): complex128,
    (uint64, complex128): complex128,
    (complex128, float16): complex128,
    (float16, complex128): complex128,
    (complex128, float32): complex128,
    (float32, complex128): complex128,
    (complex128, float64): complex128,
    (float64, complex128): complex128,
    (complex128, bfloat16): complex128,
    (bfloat16, complex128): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
}

promotion_table = {**array_api_promotion_table, **extra_promotion_table}


from .array import Array, add_ivy_array_instance_methods
from .array.conversions import *
from .array import conversions as arr_conversions
from .container import conversions as cont_conversions
from .container import (
    ContainerBase,
    Container,
    add_ivy_container_instance_methods,
)
from .nested_array import NestedArray
from .backend_handler import (
    current_backend,
    get_backend,
    set_backend,
    set_numpy_backend,
    set_jax_backend,
    set_tensorflow_backend,
    set_torch_backend,
    unset_backend,
    backend_stack,
    choose_random_backend,
    clear_backend_stack,
)
from .func_wrapper import *
from . import assertions, backend_handler, func_wrapper, exceptions
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


class GlobalsDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __name__ = dict.__name__


# defines ivy.globals attribute
globals = GlobalsDict(
    {
        "backend_stack": backend_stack,
        "default_device_stack": device.default_device_stack,
        "valid_dtypes": valid_dtypes,
        "valid_numeric_dtypes": valid_numeric_dtypes,
        "valid_int_dtypes": valid_int_dtypes,
        "valid_int_dtypes": valid_int_dtypes,
        "valid_uint_dtypes": valid_uint_dtypes,
        "valid_complex_dtypes": valid_complex_dtypes,
        "valid_devices": valid_devices,
        "invalid_dtypes": invalid_dtypes,
        "invalid_numeric_dtypes": invalid_numeric_dtypes,
        "invalid_int_dtypes": invalid_int_dtypes,
        "invalid_float_dtypes": invalid_float_dtypes,
        "invalid_uint_dtypes": invalid_uint_dtypes,
        "invalid_complex_dtypes": invalid_complex_dtypes,
        "invalid_devices": invalid_devices,
        "array_significant_figures_stack": array_significant_figures_stack,
        "array_decimal_values_stack": array_decimal_values_stack,
        "warning_level_stack": warning_level_stack,
        "queue_timeout_stack": general.queue_timeout_stack,
        "array_mode_stack": general.array_mode_stack,
        "shape_array_mode_stack": general.shape_array_mode_stack,
        "nestable_mode_stack": general.nestable_mode_stack,
        "exception_trace_mode_stack": general.exception_trace_mode_stack,
        "default_dtype_stack": data_type.default_dtype_stack,
        "default_float_dtype_stack": data_type.default_float_dtype_stack,
        "default_int_dtype_stack": data_type.default_int_dtype_stack,
        "default_uint_dtype_stack": data_type.default_uint_dtype_stack,
        "nan_policy_stack": nan_policy_stack,
    }
)


def set_global_attr(attr_name, attr_val):
    setattr(globals, attr_name, attr_val)


def del_global_attr(attr_name):
    delattr(globals, attr_name)


backend = "none"
backend_version = "none"

native_inplace_support = None

supports_gradients = None

if "IVY_BACKEND" in os.environ:
    ivy.set_backend(os.environ["IVY_BACKEND"])

# Array Significant Figures #


def _assert_array_significant_figures_formatting(sig_figs):
    ivy.assertions.check_isinstance(sig_figs, int)
    ivy.assertions.check_greater(sig_figs, 0)


# ToDo: SF formating for complex number
def _sf(x, sig_fig=3):
    if isinstance(x, np.bool_):
        return x
    if isinstance(x, complex):
        return complex(x)
    if "float" in type(x).__name__:
        x = float(
            np.format_float_positional(
                x, precision=sig_fig, unique=False, fractional=False, trim="k"
            )
        )
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
        ret = 10
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
    """Unset the currently set array significant figures."""
    global array_significant_figures_stack
    if array_significant_figures_stack:
        array_significant_figures_stack.pop(-1)


# Decimal Values #


def _assert_array_decimal_values_formatting(dec_vals):
    ivy.assertions.check_isinstance(dec_vals, int)
    ivy.assertions.check_greater(dec_vals, 0, allow_equal=True)


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
        ret = 8
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
    """Unset the currently set array decimal values."""
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
    """Unset the currently set warning level."""
    global warning_level_stack
    if warning_level_stack:
        warning_level_stack.pop(-1)


def warn(warning_message, stacklevel=0):
    warn_level = warning_level()
    warnings.filterwarnings("ignore", module=warn_to_regex[warn_level])
    warnings.warn(warning_message, stacklevel=stacklevel)


# nan policy #


def get_nan_policy():
    """Summary.

    Returns
    -------
    ret
        current nan policy, default is "nothing"

    """
    global nan_policy_stack
    if not nan_policy_stack:
        ret = "nothing"
    else:
        ret = nan_policy_stack[-1]
    return ret


def set_nan_policy(warn_level):
    """Summary.

    Parameters
    ----------
    nan_policy
        string for the nan policy to be set, one of
        "nothing", "warns", "raise_exception"

    """
    global nan_policy_stack
    if warn_level not in ["nothing", "warns", "raise_exception"]:
        raise ivy.exceptions.IvyException(
            "nan_policy must be one of 'nothing', 'warns', 'raise_exception'"
        )
    nan_policy_stack.append(warn_level)


def unset_nan_policy():
    """Unset the currently set nan policy."""
    global nan_policy_stack
    if nan_policy_stack:
        nan_policy_stack.pop(-1)
