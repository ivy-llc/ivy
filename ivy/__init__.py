# global
import copy
import re
import warnings
import builtins
import numpy as np
import sys
import inspect
import os
from collections.abc import Sequence


import ivy.utils.backend.handler
from ivy._version import __version__ as __version__

_not_imported_backends = list(ivy.utils.backend.handler._backend_dict.keys())
try:
    # Skip numpy from frameworks installed
    _not_imported_backends.remove("numpy")
except KeyError:
    pass
for backend_framework in _not_imported_backends.copy():
    # If a framework was already imported before our init execution
    if backend_framework in sys.modules:
        _not_imported_backends.remove(backend_framework)

warnings.filterwarnings("ignore", module="^(?!.*ivy).*$")


# Local Ivy

import_module_path = "ivy.utils._importlib"


def is_local():
    return hasattr(ivy, "_is_local_pkg")


# class placeholders


class FrameworkStr(str):
    def __new__(cls, fw_str):
        ivy.utils.assertions.check_elem_in_list(
            fw_str, ivy.utils.backend.handler._backend_dict.keys()
        )
        return str.__new__(cls, fw_str)


class Framework:
    pass


class NativeArray:
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
            ivy.utils.assertions.check_elem_in_list(dev_str[0:3], ["gpu", "tpu", "cpu"])
            if dev_str != "cpu":
                # ivy.assertions.check_equal(dev_str[3], ":")
                ivy.utils.assertions.check_true(
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
            raise ivy.utils.exceptions.IvyException("dtype must be type str")
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.utils.exceptions.IvyException(
                f"{dtype_str} is not supported by ivy"
            )
        return str.__new__(cls, dtype_str)

    def __ge__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self == ivy.promote_types(self, other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self >= other and self != other

    def __lt__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.utils.exceptions.IvyException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            )

        return self != ivy.promote_types(self, other)

    def __le__(self, other):
        if isinstance(other, str):
            other = Dtype(other)

        if not isinstance(other, Dtype):
            raise ivy.utils.exceptions.IvyException(
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
    def name(self) -> str:
        return str(self)

    @property
    def info(self):
        if self.is_int_dtype or self.is_uint_dtype:
            return iinfo(self)
        elif self.is_float_dtype:
            return finfo(self)
        else:
            raise ivy.utils.exceptions.IvyError(f"{self} is not supported by info")

    def can_cast(self, to):
        return can_cast(self, to)


class Shape(Sequence):
    def __init__(self, shape_tup):
        valid_types = (int, list, tuple, ivy.Array, ivy.Shape)
        if len(backend_stack) != 0:
            valid_types += (ivy.NativeShape, ivy.NativeArray)
        else:
            valid_types += (
                current_backend(shape_tup).NativeShape,
                current_backend(shape_tup).NativeArray,
            )
        ivy.utils.assertions.check_isinstance(shape_tup, valid_types)
        if len(backend_stack) == 0:
            if isinstance(shape_tup, np.ndarray):
                shape_tup = tuple(shape_tup.tolist())
            self._shape = shape_tup
        elif isinstance(shape_tup, valid_types):
            self._shape = ivy.to_native_shape(shape_tup)
        else:
            self._shape = None

    @staticmethod
    def _shape_casting_helper(ivy_shape, other):
        if isinstance(other, tuple) and not isinstance(ivy_shape, tuple):
            return tuple(ivy_shape)
        elif isinstance(other, list) and not isinstance(ivy_shape, list):
            return list(ivy_shape)
        else:
            return ivy_shape

    def __repr__(self):
        pattern = r"\d+(?:,\s*\d+)*"
        shape_repr = re.findall(pattern, self._shape.__str__())
        shape_repr = ", ".join([str(i) for i in shape_repr])
        shape_repr = shape_repr + "," if len(shape_repr) == 1 else shape_repr
        return (
            f"ivy.Shape({shape_repr})" if self._shape is not None else "ivy.Shape(None)"
        )

    def __iter__(self):
        return iter(self._shape)

    def __add__(self, other):
        try:
            self._shape = self._shape + other
        except TypeError:
            self._shape = self._shape + list(other)
        return self

    def __radd__(self, other):
        try:
            self._shape = other + self._shape
        except TypeError:
            self._shape = list(other) + self._shape
        return self

    def __mul__(self, other):
        self._shape = self._shape * other
        return self

    def __rmul__(self, other):
        self._shape = other * self._shape
        return self

    def __bool__(self):
        return self._shape.__bool__()

    def __div__(self, other):
        return self._shape // other

    def __floordiv__(self, other):
        return self._shape // other

    def __mod__(self, other):
        return self._shape % other

    def __rdiv__(self, other):
        return other // self._shape

    def __rmod__(self, other):
        return other % self._shape

    def __reduce__(self):
        return (self._shape,)

    def as_dimension(self, other):
        if isinstance(other, self._shape):
            return other
        else:
            return self._shape

    def __sub__(self, other):
        try:
            self._shape = self._shape - other
        except TypeError:
            self._shape = self._shape - list(other)
        return self

    def __rsub__(self, other):
        try:
            self._shape = other - self._shape
        except TypeError:
            self._shape = list(other) - self._shape
        return self

    def __eq__(self, other):
        self._shape = Shape._shape_casting_helper(self._shape, other)
        return self._shape == other

    def __int__(self):
        if hasattr(self._shape, "__int__"):
            res = self._shape.__int__()
        else:
            res = int(self._shape)
        if res is NotImplemented:
            return res
        return to_ivy(res)

    def __ge__(self, other):
        self._shape = Shape._shape_casting_helper(self._shape, other)
        return self._shape >= other

    def __gt__(self, other):
        self._shape = Shape._shape_casting_helper(self._shape, other)
        return self._shape > other

    def __le__(self, other):
        self._shape = Shape._shape_casting_helper(self._shape, other)
        return self._shape <= other

    def __lt__(self, other):
        self._shape = Shape._shape_casting_helper(self._shape, other)
        return self._shape < other

    def __getattribute__(self, item):
        return super().__getattribute__(item)

    def __getitem__(self, key):
        try:
            return self._shape[key]
        except (TypeError, IndexError):
            return None

    def __len__(self):
        return len(self._shape) if self._shape is not None else 0

    def __delattr__(self, item):
        return super().__delattr__(item)

    def __hash__(self):
        return hash(self._shape)

    def __sizeof__(self):
        return len(self._shape) if self._shape is not None else 0

    def __dir__(self):
        return self._shape.__dir__()

    def __pow__(self, power, modulo=None):
        pass

    def __index__(self):
        pass

    def __rdivmod__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __ne__(self, other):
        pass

    @property
    def shape(self):
        return self._shape

    @property
    def value(self):
        return self._value

    def concatenate(self, other):
        if self._shape is None or other.dims is None:
            raise ValueError("Unknown Shape")
        else:
            return Shape(self.dims + other.dims)

    def index(self, index):
        assert isinstance(self._shape, Shape)
        if self._shape.rank is None:
            return Shape(None)
        else:
            return self._shape[index]

    @property
    def shape(self):
        return self._shape

    def as_dimension(self):
        if isinstance(self._shape, Shape):
            return self._shape
        else:
            return Shape(self._shape)

    def is_compatible_with(self, other):
        return self._shape is None or other.value is None or self._shape == other.value

    @property
    def rank(self):
        """Returns the rank of this shape, or None if it is unspecified."""
        if self._shape is not None:
            return len(self._shape)
        return None

    def assert_same_rank(self, other):
        other = Shape(other)
        if self.rank != other.rank:
            raise ValueError("Shapes %s and %s must have the same rank" % (self, other))

    def assert_has_rank(self, rank):
        if self.rank not in (None, rank):
            raise ValueError("Shape %s must have rank %d" % (self, rank))

    def unknown_shape(rank=None, **kwargs):
        if rank is None and "ndims" in kwargs:
            rank = kwargs.pop("ndims")
        if kwargs:
            raise TypeError("Unknown argument: %s" % kwargs)
        if rank is None:
            return Shape(None)
        else:
            return Shape([Shape(None)] * rank)

    def with_rank(self, rank):
        try:
            return self.merge_with(unknown_shape(rank=rank))
        except ValueError:
            raise ValueError("Shape %s must have rank %d" % (self, rank))

    def with_rank_at_least(self, rank):
        if self.rank is not None and self.rank < rank:
            raise ValueError("Shape %s must have rank at least %d" % (self, rank))
        else:
            return self

    def with_rank_at_most(self, rank):
        if self.rank is not None and self.rank > rank:
            raise ValueError("Shape %s must have rank at most %d" % (self, rank))
        else:
            return self

    def as_shape(shape):
        if isinstance(shape, Shape):
            return shape
        else:
            return Shape(shape)

    @property
    def dims(self):
        if self._shape is None:
            return None
        # return [as_dimension(d) for d in self._shape]

    @property
    def ndims(self):
        """Deprecated accessor for `rank`."""
        return self.rank

    @property
    def is_fully_defined(self):
        return self._shape is not None and all(
            shape is not None for shape in self._shape
        )

    property

    def num_elements(self):
        if not self.is_fully_defined():
            return None

    @property
    def assert_is_fully_defined(self):
        if not self.is_fully_defined():
            raise ValueError("Shape %s is not fully defined" % self)

    def as_list(self):
        if self._shape is None:
            raise ivy.utils.exceptions.IvyException(
                "Cannot convert a partially known Shape to a list"
            )
        return [dim for dim in self._shape]


class IntDtype(Dtype):
    def __new__(cls, dtype_str):
        if dtype_str is builtins.int:
            dtype_str = default_int_dtype()
        if not isinstance(dtype_str, str):
            raise ivy.utils.exceptions.IvyException("dtype_str must be type str")
        if "int" not in dtype_str:
            raise ivy.utils.exceptions.IvyException(
                "dtype must be string and starts with int"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.utils.exceptions.IvyException(
                f"{dtype_str} is not supported by ivy"
            )
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return iinfo(self)


class FloatDtype(Dtype):
    def __new__(cls, dtype_str):
        if dtype_str is builtins.float:
            dtype_str = default_float_dtype()
        if not isinstance(dtype_str, str):
            raise ivy.utils.exceptions.IvyException("dtype_str must be type str")
        if "float" not in dtype_str:
            raise ivy.utils.exceptions.IvyException(
                "dtype must be string and starts with float"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.utils.exceptions.IvyException(
                f"{dtype_str} is not supported by ivy"
            )
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return finfo(self)


class UintDtype(IntDtype):
    def __new__(cls, dtype_str):
        if not isinstance(dtype_str, str):
            raise ivy.utils.exceptions.IvyException("dtype_str must be type str")
        if "uint" not in dtype_str:
            raise ivy.utils.exceptions.IvyException(
                "dtype must be string and starts with uint"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.utils.exceptions.IvyException(
                f"{dtype_str} is not supported by ivy"
            )
        return str.__new__(cls, dtype_str)

    @property
    def info(self):
        return iinfo(self)


class ComplexDtype(Dtype):
    def __new__(cls, dtype_str):
        if not isinstance(dtype_str, str):
            raise ivy.utils.exceptions.IvyException("dtype_str must be type str")
        if "complex" not in dtype_str:
            raise ivy.utils.exceptions.IvyException(
                "dtype must be string and starts with complex"
            )
        if dtype_str not in _all_ivy_dtypes_str:
            raise ivy.utils.exceptions.IvyException(
                f"{dtype_str} is not supported by ivy"
            )
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
dynamic_backend_stack = list()
warn_to_regex = {"all": "!.*", "ivy_only": "^(?!.*ivy).*$", "none": ".*"}


# local
import threading


# devices
# ToDo: add gpu and tpu for valid devices when we test for them
all_devices = ("cpu", "gpu", "tpu")

valid_devices = ("cpu", "gpu")

invalid_devices = ("tpu",)

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
    complex64,
    complex128,
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

locks = {"backend_setter": threading.Lock()}


from .func_wrapper import *
from .data_classes.array import Array, add_ivy_array_instance_methods
from .data_classes.array.conversions import *
from .data_classes.array import conversions as arr_conversions
from .data_classes.container import conversions as cont_conversions
from .data_classes.container import (
    ContainerBase,
    Container,
    add_ivy_container_instance_methods,
)
from .data_classes.nested_array import NestedArray
from ivy.utils.backend import (
    current_backend,
    compiled_backends,
    with_backend,
    set_backend,
    set_numpy_backend,
    set_jax_backend,
    set_tensorflow_backend,
    set_torch_backend,
    set_paddle_backend,
    set_mxnet_backend,
    previous_backend,
    backend_stack,
    choose_random_backend,
    unset_backend,
)
from . import func_wrapper
from .utils import assertions, exceptions, verbosity
from .utils.backend import handler
from . import functional
from .functional import *
from . import stateful
from .stateful import *
from ivy.utils.inspection import fn_array_spec, add_array_specs

add_array_specs()

_imported_frameworks_before_compiler = list(sys.modules.keys())
try:
    from .compiler.compiler import transpile, compile, unify
except:  # noqa: E722
    pass  # Added for the finally statment
finally:
    # Skip framework imports done by Ivy compiler for now
    for backend_framework in _not_imported_backends.copy():
        if backend_framework in sys.modules:
            if backend_framework not in _imported_frameworks_before_compiler:
                _not_imported_backends.remove(backend_framework)


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

    def __deepcopy__(self, memo):
        ret = self.__class__.__new__(self.__class__)
        for k, v in self.items():
            ret[k] = copy.deepcopy(v)
        return ret


# defines ivy.globals attribute
globals_vars = GlobalsDict(
    {
        "backend_stack": backend_stack,
        "default_device_stack": device.default_device_stack,
        "valid_dtypes": valid_dtypes,
        "valid_numeric_dtypes": valid_numeric_dtypes,
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
        "soft_device_mode_stack": device.soft_device_mode_stack,
        "shape_array_mode_stack": general.shape_array_mode_stack,
        "show_func_wrapper_trace_mode_stack": (
            general.show_func_wrapper_trace_mode_stack
        ),
        "min_denominator_stack": general.min_denominator_stack,
        "min_base_stack": general.min_base_stack,
        "tmp_dir_stack": general.tmp_dir_stack,
        "precise_mode_stack": general.precise_mode_stack,
        "nestable_mode_stack": general.nestable_mode_stack,
        "exception_trace_mode_stack": general.exception_trace_mode_stack,
        "default_dtype_stack": data_type.default_dtype_stack,
        "default_float_dtype_stack": data_type.default_float_dtype_stack,
        "default_int_dtype_stack": data_type.default_int_dtype_stack,
        "default_uint_dtype_stack": data_type.default_uint_dtype_stack,
        "nan_policy_stack": nan_policy_stack,
        "dynamic_backend_stack": dynamic_backend_stack,
    }
)

_default_globals = copy.deepcopy(globals_vars)


def reset_globals():
    global globals_vars
    globals_vars = copy.deepcopy(_default_globals)


def set_global_attr(attr_name, attr_val):
    setattr(globals_vars, attr_name, attr_val)


def del_global_attr(attr_name):
    delattr(globals_vars, attr_name)


backend = ""
backend_version = {}

native_inplace_support = None

supports_gradients = None


# Array Significant Figures #


def _assert_array_significant_figures_formatting(sig_figs):
    ivy.utils.assertions.check_isinstance(sig_figs, int)
    ivy.utils.assertions.check_greater(sig_figs, 0, as_array=False)


# ToDo: SF formating for complex number
def vec_sig_fig(x, sig_fig=3):
    if isinstance(x, np.bool_):
        return x
    if isinstance(x, complex):
        return complex(x)
    if np.issubdtype(x.dtype, np.floating):
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sig_fig - 1))
        mags = 10 ** (sig_fig - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags
    return x


ivy.array_significant_figures = (
    array_significant_figures_stack[-1] if array_significant_figures_stack else 10
)


def set_array_significant_figures(sig_figs):
    """
    Summary.

    Parameters
    ----------
    sig_figs
        optional int, number of significant figures to be shown when printing
    """
    _assert_array_significant_figures_formatting(sig_figs)
    global array_significant_figures_stack
    array_significant_figures_stack.append(sig_figs)
    ivy.__setattr__("array_significant_figures", sig_figs, True)


def unset_array_significant_figures():
    """Unset the currently set array significant figures."""
    global array_significant_figures_stack
    if array_significant_figures_stack:
        array_significant_figures_stack.pop(-1)
        sig_figs = (
            array_significant_figures_stack[-1]
            if array_significant_figures_stack
            else 10
        )
        ivy.__setattr__("array_significant_figures", sig_figs, True)


# Decimal Values #


def _assert_array_decimal_values_formatting(dec_vals):
    ivy.utils.assertions.check_isinstance(dec_vals, int)
    ivy.utils.assertions.check_greater(dec_vals, 0, allow_equal=True, as_array=False)


ivy.array_decimal_values = (
    array_decimal_values_stack[-1] if array_decimal_values_stack else 8
)


def set_array_decimal_values(dec_vals):
    """
    Summary.

    Parameters
    ----------
    dec_vals
        optional int, number of significant figures to be shown when printing
    """
    _assert_array_decimal_values_formatting(dec_vals)
    global array_decimal_values_stack
    array_decimal_values_stack.append(dec_vals)
    ivy.__setattr__("array_decimal_values", dec_vals, True)


def unset_array_decimal_values():
    """Unset the currently set array decimal values."""
    global array_decimal_values_stack
    if array_decimal_values_stack:
        array_decimal_values_stack.pop(-1)
        dec_vals = array_decimal_values_stack[-1] if array_decimal_values_stack else 8
        ivy.__setattr__("array_decimal_values", dec_vals, True)


ivy.warning_level = warning_level_stack[-1] if warning_level_stack else "ivy_only"


def set_warning_level(warn_level):
    """
    Summary.

    Parameters
    ----------
    warn_level
        string for the warning level to be set, one of "none", "ivy_only", "all"
    """
    global warning_level_stack
    warning_level_stack.append(warn_level)
    ivy.__setattr__("warning_level", warn_level, True)


def unset_warning_level():
    """Unset the currently set warning level."""
    global warning_level_stack
    if warning_level_stack:
        warning_level_stack.pop(-1)
        warn_level = warning_level_stack[-1] if warning_level_stack else "ivy_only"
        ivy.__setattr__("warning_level", warn_level, True)


def warn(warning_message, stacklevel=0):
    warn_level = ivy.warning_level
    warnings.filterwarnings("ignore", module=warn_to_regex[warn_level])
    warnings.warn(warning_message, stacklevel=stacklevel)


# nan policy #
ivy.nan_policy = nan_policy_stack[-1] if nan_policy_stack else "nothing"


def set_nan_policy(warn_level):
    """
    Summary.

    Parameters
    ----------
    nan_policy
        string for the nan policy to be set, one of
        "nothing", "warns", "raise_exception"
    """
    if warn_level not in ["nothing", "warns", "raise_exception"]:
        raise ivy.utils.exceptions.IvyException(
            "nan_policy must be one of 'nothing', 'warns', 'raise_exception'"
        )
    global nan_policy_stack
    nan_policy_stack.append(warn_level)
    ivy.__setattr__("nan_policy", warn_level, True)


def unset_nan_policy():
    """Unset the currently set nan policy."""
    global nan_policy_stack
    if nan_policy_stack:
        nan_policy_stack.pop(-1)
        warn_level = nan_policy_stack[-1] if nan_policy_stack else "nothing"
        ivy.__setattr__("nan_policy", warn_level, True)


# Dynamic Backend


ivy.dynamic_backend = dynamic_backend_stack[-1] if dynamic_backend_stack else True


def set_dynamic_backend(flag):
    """Set the global dynamic backend setting to the provided flag (True or False)"""
    global dynamic_backend_stack
    if flag not in [True, False]:
        raise ValueError("dynamic_backend must be a boolean value (True or False)")
    dynamic_backend_stack.append(flag)
    ivy.__setattr__("dynamic_backend", flag, True)


def unset_dynamic_backend():
    """
    Remove the current dynamic backend setting.

    Also restore the previous setting (if any)
    """
    global dynamic_backend_stack
    if dynamic_backend_stack:
        dynamic_backend_stack.pop()
        flag = dynamic_backend_stack[-1] if dynamic_backend_stack else True
        ivy.__setattr__("dynamic_backend", flag, True)


# Context Managers


class DynamicBackendContext:
    def __init__(self, value):
        self.value = value
        self.original = None

    def __enter__(self):
        self.original = ivy.dynamic_backend
        set_dynamic_backend(self.value)

    def __exit__(self, type, value, traceback):
        unset_dynamic_backend()
        if self.original is not None:
            set_dynamic_backend(self.original)


def dynamic_backend_as(value):
    return DynamicBackendContext(value)


for backend_framework in _not_imported_backends:
    if backend_framework in sys.modules:
        warnings.warn(
            f"{backend_framework} module has been imported while ivy doesn't "
            "import it without setting a backend, ignore if that's intended"
        )


# sub_backends
from ivy.utils.backend.sub_backend_handler import (
    set_sub_backend,
    unset_sub_backend,
    clear_sub_backends,
    available_sub_backends,
)


def current_sub_backends():
    return []


# casting modes

downcast_dtypes = False
upcast_dtypes = False
crosscast_dtypes = False
cast_dtypes = lambda: downcast_dtypes and upcast_dtypes and crosscast_dtypes


def downcast_data_types(val=True):
    global downcast_dtypes
    downcast_dtypes = val


def upcast_data_types(val=True):
    global upcast_dtypes
    upcast_dtypes = val


def crosscast_data_types(val=True):
    global crosscast_dtypes
    crosscast_dtypes = val


def cast_data_types(val=True):
    global upcast_dtypes
    global downcast_dtypes
    global crosscast_dtypes
    upcast_dtypes = val
    downcast_dtypes = val
    crosscast_dtypes = val


# Promotion Tables #
# ---------------- #


# data type promotion
array_api_promotion_table = {
    (bool, bool): bool,
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, int64): int64,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, uint64): uint64,
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, float64): float64,
}

# the extra promotion table follows numpy safe casting convention
# the following link discusses the different approaches to dtype promotions
# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
common_extra_promotion_table = {
    (bool, int8): int8,
    (bool, int16): int16,
    (bool, int32): int32,
    (bool, int64): int64,
    (bool, uint8): uint8,
    (bool, uint16): uint16,
    (bool, uint32): uint32,
    (bool, uint64): uint64,
    (bool, float16): float16,
    (bool, float32): float32,
    (bool, float64): float64,
    (bool, bfloat16): bfloat16,
    (bool, complex64): complex64,
    (bool, complex128): complex128,
    (int8, float16): float16,
    (int8, float32): float32,
    (int8, float64): float64,
    (int8, bfloat16): bfloat16,
    (int8, complex64): complex64,
    (int8, complex128): complex128,
    (int16, float32): float32,
    (int16, float64): float64,
    (int16, complex64): complex64,
    (int16, complex128): complex128,
    (int32, float64): float64,
    (int32, complex128): complex128,
    (int64, float64): float64,
    (int64, complex128): complex128,
    (uint8, float16): float16,
    (uint8, float32): float32,
    (uint8, float64): float64,
    (uint8, bfloat16): bfloat16,
    (uint8, complex64): complex64,
    (uint8, complex128): complex128,
    (uint16, float32): float32,
    (uint16, float64): float64,
    (uint16, complex64): complex64,
    (uint16, complex128): complex128,
    (uint32, float64): float64,
    (uint32, complex128): complex128,
    (uint64, int8): float64,
    (uint64, int16): float64,
    (uint64, int32): float64,
    (uint64, int64): float64,
    (uint64, float64): float64,
    (uint64, complex128): complex128,
    (float16, bfloat16): float32,
    (float16, complex64): complex64,
    (float16, complex128): complex128,
    (float32, complex64): complex64,
    (float32, complex128): complex128,
    (float64, complex64): complex128,
    (float64, complex128): complex128,
    (bfloat16, float16): float32,
    (bfloat16, float32): float32,
    (bfloat16, float64): float64,
    (bfloat16, bfloat16): bfloat16,
    (bfloat16, complex64): complex64,
    (bfloat16, complex128): complex128,
    (complex64, float64): complex128,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, complex128): complex128,
}
# Avoiding All Precision Loss (Numpy Approach)
precise_extra_promotion_table = {
    (float16, int16): float32,
    (float16, int32): float64,
    (float16, int64): float64,
    (float16, uint16): float32,
    (float16, uint32): float64,
    (float16, uint64): float64,
    (float32, int32): float64,
    (float32, int64): float64,
    (float32, uint32): float64,
    (float32, uint64): float64,
    (bfloat16, int16): float32,
    (bfloat16, int32): float64,
    (bfloat16, int64): float64,
    (bfloat16, uint16): float32,
    (bfloat16, uint32): float64,
    (bfloat16, uint64): float64,
    (complex64, int32): complex128,
    (complex64, int64): complex128,
    (complex64, uint32): complex128,
    (complex64, uint64): complex128,
}

extra_promotion_table = {
    (float16, int16): float16,
    (float16, int32): float16,
    (float16, int64): float16,
    (float16, uint16): float16,
    (float16, uint32): float16,
    (float16, uint64): float16,
    (float32, int32): float32,
    (float32, int64): float32,
    (float32, uint32): float32,
    (float32, uint64): float32,
    (bfloat16, int16): bfloat16,
    (bfloat16, int32): bfloat16,
    (bfloat16, int64): bfloat16,
    (bfloat16, uint16): bfloat16,
    (bfloat16, uint32): bfloat16,
    (bfloat16, uint64): bfloat16,
    (complex64, int32): complex64,
    (complex64, int64): complex64,
    (complex64, uint32): complex64,
    (complex64, uint64): complex64,
}

# TODO: change when it's not the default mode anymore
promotion_table = {
    **array_api_promotion_table,
    **common_extra_promotion_table,
    **precise_extra_promotion_table,
}


# global parameter properties
GLOBAL_PROPS = [
    "array_significant_figures",
    "array_decimal_values",
    "warning_level",
    "nan_policy",
    "array_mode",
    "nestable_mode",
    "exception_trace_mode",
    "show_func_wrapper_trace_mode",
    "min_denominator",
    "min_base",
    "queue_timeout",
    "tmp_dir",
    "shape_array_mode",
    "dynamic_backend",
    "precise_mode",
    "soft_device_mode",
    "logging_mode"
    "default_dtype",
    "default_float_dtype",
    "default_int_dtype",
    "default_complex_dtype",
    "default_uint_dtype",
    "global_attr",
    "jax_backend",
    "mxnet_backend",
    "numpy_backend",
    "paddle_backend",
    "tensorflow_backend",
    "torch_backend",
    "nest_at_index",
    "nest_at_indices",
    "split_factor",
    "sub_backend",
]


INTERNAL_FILENAMES = [
    os.path.join("ivy", "compiler"),
    os.path.join("ivy", "functional"),
    os.path.join("ivy", "data_classes"),
    os.path.join("ivy", "stateful"),
    os.path.join("ivy", "utils"),
    os.path.join("ivy_tests", "test_ivy"),
    os.path.join("ivy", "func_wrapper.py"),
    os.path.join("ivy", "__init__.py"),
]


def _is_from_internal(filename):
    return builtins.any([fn in filename for fn in INTERNAL_FILENAMES])


class LoggingMode:
    logging_modes = ["DEBUG", "INFO", "WARNING", "ERROR"]
    logging_mode_stack = []

    def __init__(self):
        # Set up the initial logging mode
        logging.basicConfig(level=logging.WARNING)
        self.logging_mode_stack.append(logging.WARNING)

    def set_logging_mode(self, mode):
        """
        Set the current logging mode for Ivy.

        Possible modes are 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        """
        assert (
            mode in self.logging_modes
        ), "Invalid logging mode. Choose from: " + ", ".join(self.logging_modes)

        # Update the logging level
        logging.getLogger().setLevel(mode)
        self.logging_mode_stack.append(mode)

    def unset_logging_mode(self):
        """Remove the most recently set logging mode, returning to the previous one."""
        if len(self.logging_mode_stack) > 1:
            # Remove the current mode
            self.logging_mode_stack.pop()

            # Set the previous mode
            logging.getLogger().setLevel(self.logging_mode_stack[-1])


class IvyWithGlobalProps(sys.modules[__name__].__class__):
    def __setattr__(self, name, value, internal=False):
        previous_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(previous_frame)[0]
        internal = internal and _is_from_internal(filename)
        if not internal and name in GLOBAL_PROPS:
            raise ivy.utils.exceptions.IvyException(
                "Property: {} is read only! Please use the setter: set_{}() for setting"
                " its value!".format(name, name)
            )
        self.__dict__[name] = value


if (
    "ivy" in sys.modules.keys()
    and sys.modules["ivy"].utils._importlib.IS_COMPILING_WITH_BACKEND
):
    # Required for ivy.with_backend internal compilation
    sys.modules["ivy"].utils._importlib.import_cache[
        __name__
    ].__class__ = IvyWithGlobalProps
else:
    sys.modules[__name__].__class__ = IvyWithGlobalProps
