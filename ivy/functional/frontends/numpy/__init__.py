# flake8: noqa
import ivy
from ivy.utils.exceptions import handle_exceptions
from typing import Union, Iterable, Tuple
from numbers import Number
from .data_type_routines import dtype
from . import ndarray
from .ndarray import *
from . import scalars
from .scalars import *

# Constructing dtypes are required as ivy.<dtype>
# will change dynamically on the backend and may not be available
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.UintDtype("uint8")
uint16 = ivy.UintDtype("uint16")
uint32 = ivy.UintDtype("uint32")
uint64 = ivy.UintDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
complex64 = ivy.ComplexDtype("complex64")
complex128 = ivy.ComplexDtype("complex128")
bool = ivy.Dtype("bool")

numpy_promotion_table = {
    (bool, bool): bool,
    (bool, int8): int8,
    (bool, int16): int16,
    (bool, int32): int32,
    (bool, int64): int64,
    (bool, uint8): uint8,
    (bool, uint16): uint16,
    (bool, uint32): uint32,
    (bool, uint64): uint64,
    (bool, bfloat16): bfloat16,
    (bool, float16): float16,
    (bool, float32): float32,
    (bool, float64): float64,
    (bool, complex64): complex64,
    (bool, complex128): complex128,
    (bool, bool): bool,
    (int8, bool): int8,
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int16, bool): int16,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int32, bool): int32,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int64, bool): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (uint8, bool): uint8,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint16, bool): uint16,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint32, bool): uint32,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint64, bool): uint64,
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
    (float16, bool): float16,
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, bool): float32,
    (float32, float16): float32,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, bool): float64,
    (float64, float16): float64,
    (float64, float32): float64,
    (float64, float64): float64,
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
    (int16, float16): float32,
    (float16, int16): float32,
    (int16, float32): float32,
    (float32, int16): float32,
    (int16, float64): float64,
    (float64, int16): float64,
    (int32, float16): float64,
    (float16, int32): float64,
    (int32, float32): float64,
    (float32, int32): float64,
    (int32, float64): float64,
    (float64, int32): float64,
    (int64, float16): float64,
    (float16, int64): float64,
    (int64, float32): float64,
    (float32, int64): float64,
    (int64, float64): float64,
    (float64, int64): float64,
    (uint8, float16): float16,
    (float16, uint8): float16,
    (uint8, float32): float32,
    (float32, uint8): float32,
    (uint8, float64): float64,
    (float64, uint8): float64,
    (uint16, float16): float32,
    (float16, uint16): float32,
    (uint16, float32): float32,
    (float32, uint16): float32,
    (uint16, float64): float64,
    (float64, uint16): float64,
    (uint32, float16): float64,
    (float16, uint32): float64,
    (uint32, float32): float64,
    (float32, uint32): float64,
    (uint32, float64): float64,
    (float64, uint32): float64,
    (uint64, float16): float64,
    (float16, uint64): float64,
    (uint64, float32): float64,
    (float32, uint64): float64,
    (uint64, float64): float64,
    (float64, uint64): float64,
    (bfloat16, bfloat16): bfloat16,
    (bfloat16, uint8): bfloat16,
    (uint8, bfloat16): bfloat16,
    (bfloat16, int8): bfloat16,
    (int8, bfloat16): bfloat16,
    (bfloat16, float32): float32,
    (float32, bfloat16): float32,
    (bfloat16, float64): float64,
    (float64, bfloat16): float64,
    (complex64, bool): complex64,
    (complex64, int8): complex64,
    (complex64, int16): complex64,
    (complex64, int32): complex128,
    (complex64, int64): complex128,
    (complex64, uint8): complex64,
    (complex64, uint16): complex64,
    (complex64, uint32): complex128,
    (complex64, uint64): complex128,
    (complex64, float16): complex64,
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex64, bfloat16): complex64,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, bool): complex128,
    (complex128, int8): complex128,
    (complex128, int16): complex128,
    (complex128, int32): complex128,
    (complex128, int64): complex128,
    (complex128, uint8): complex128,
    (complex128, uint16): complex128,
    (complex128, uint32): complex128,
    (complex128, uint64): complex128,
    (complex128, float16): complex128,
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (complex128, bfloat16): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
    (int8, complex64): complex64,
    (int16, complex64): complex64,
    (int32, complex64): complex128,
    (int64, complex64): complex128,
    (uint8, complex64): complex64,
    (uint16, complex64): complex64,
    (uint32, complex64): complex128,
    (uint64, complex64): complex128,
    (float16, complex64): complex64,
    (float32, complex64): complex64,
    (float64, complex64): complex128,
    (bfloat16, complex64): complex64,
    (int8, complex128): complex128,
    (int16, complex128): complex128,
    (int32, complex128): complex128,
    (int64, complex128): complex128,
    (uint8, complex128): complex128,
    (uint16, complex128): complex128,
    (uint32, complex128): complex128,
    (uint64, complex128): complex128,
    (float16, complex128): complex128,
    (float32, complex128): complex128,
    (float64, complex128): complex128,
    (bfloat16, complex128): complex128,
}

numpy_str_to_type_table = {
    "b": "int8",
    "h": "int16",
    "i": "int32",
    "l": "int64",
    "q": "int64",
    "B": "uint8",
    "H": "uint16",
    "I": "uint32",
    "L": "uint64",
    "e": "float16",
    "f": "float32",
    "d": "float64",
    "?": "bool",
    "E": "bfloat16",
    "F": "complex64",
    "D": "complex128",
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "i1": "int8",
    "i2": "int16",
    "i4": "int32",
    "i8": "int64",
    "u1": "uint8",
    "u2": "uint16",
    "u4": "uint32",
    "u8": "uint64",
    "c8": "complex64",
    "c16": "complex128",
    "bool_": "bool",
}

numpy_type_to_str_and_num_table = {
    "int8": ("b", 1),
    "int16": ("h", 3),
    "int32": ("i", 5),
    "int64": ("l", 7),
    "uint8": ("B", 2),
    "uint16": ("H", 4),
    "uint32": ("I", 6),
    "uint64": ("L", 8),
    "float16": ("e", 23),
    "float32": ("f", 11),
    "float64": ("d", 12),
    "bool": ("?", 0),
    "bfloat16": ("E", 256),
    "complex64": ("F", 14),
    "complex128": ("D", 15),
}

numpy_scalar_to_dtype = {
    bool_: bool,
    number: float64,
    integer: int64,
    signedinteger: int64,
    byte: int8,
    short: int16,
    intc: int32,
    longlong: int64,
    int_: int64,
    unsignedinteger: uint64,
    ubyte: uint8,
    ushort: uint16,
    uintc: uint32,
    ulonglong: uint64,
    uint: uint64,
    inexact: float64,
    floating: float64,
    half: float16,
    single: float32,
    float_: float64,
    bfloat16: bfloat16,
    complexfloating: complex128,
    csingle: complex64,
    complex_: complex128,
}

numpy_dtype_to_scalar = {v: k for k, v in numpy_scalar_to_dtype.items()}

numpy_casting_rules = {
    bool: [
        bool,
        uint8,
        uint16,
        uint32,
        uint64,
        int8,
        int16,
        int32,
        int64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
    ],
    int8: [
        int8,
        int16,
        int32,
        int64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
    ],
    int16: [
        int16,
        int32,
        int64,
        float32,
        float64,
        complex64,
        complex128,
    ],
    int32: [int32, int64, float64, complex128],
    int64: [int64, float64, complex128],
    uint8: [
        uint8,
        uint16,
        uint32,
        uint64,
        int16,
        int32,
        int64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
    ],
    uint16: [
        uint16,
        uint32,
        uint64,
        int32,
        int64,
        float32,
        float64,
        complex64,
        complex128,
    ],
    uint32: [
        uint32,
        uint64,
        int64,
        float64,
        complex128,
    ],
    uint64: [uint64, float64, complex128],
    float16: [
        float16,
        float32,
        float64,
        complex64,
        complex128,
    ],
    float32: [
        float32,
        float64,
        complex64,
        complex128,
    ],
    float64: [float64, complex128],
    complex64: [complex64, complex128],
    complex128: [complex128],
}


@handle_exceptions
def promote_numpy_dtypes(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
):
    type1, type2 = ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2)
    try:
        return numpy_promotion_table[(type1, type2)]
    except KeyError:
        raise ivy.utils.exceptions.IvyException("these dtypes are not type promotable")


@handle_exceptions
def promote_types_of_numpy_inputs(
    x1: Union[ivy.Array, Number, Iterable[Number]],
    x2: Union[ivy.Array, Number, Iterable[Number]],
    /,
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Promotes the dtype of the given ivy array inputs to a common dtype
    based on numpy type promotion rules. While passing float or integer values or any
    other non-array input to this function, it should be noted that the return will
    be an array-like object. Therefore, outputs from this function should be used
    as inputs only for those functions that expect an array-like or tensor-like objects,
    otherwise it might give unexpected results.
    """
    # Ignore type of 0-dim arrays to mimic numpy
    if (
        hasattr(x1, "shape")
        and x1.shape == ()
        and not (hasattr(x2, "shape") and x2.shape == ())
    ):
        x1 = ivy.to_scalar(x1)
    if (
        hasattr(x2, "shape")
        and x2.shape == ()
        and not (hasattr(x1, "shape") and x1.shape == ())
    ):
        x2 = ivy.to_scalar(x2)
    type1 = ivy.default_dtype(item=x1).strip("u123456789")
    type2 = ivy.default_dtype(item=x2).strip("u123456789")
    if hasattr(x1, "dtype") and not hasattr(x2, "dtype") and type1 == type2:
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(
            x2, dtype=x1.dtype, device=ivy.default_device(item=x1, as_native=False)
        )
    elif not hasattr(x1, "dtype") and hasattr(x2, "dtype") and type1 == type2:
        x1 = ivy.asarray(
            x1, dtype=x2.dtype, device=ivy.default_device(item=x2, as_native=False)
        )
        x2 = ivy.asarray(x2)
    else:
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2)
        promoted = promote_numpy_dtypes(x1.dtype, x2.dtype)
        x1 = ivy.asarray(x1, dtype=promoted)
        x2 = ivy.asarray(x2, dtype=promoted)
    return x1, x2


from . import creation_routines
from .creation_routines import *
from . import data_type_routines
from .data_type_routines import *
from . import indexing_routines
from .indexing_routines import *
from . import logic
from .logic import *
from . import manipulation_routines
from .manipulation_routines import *
from . import mathematical_functions
from .mathematical_functions import *
from . import sorting_searching_counting
from .sorting_searching_counting import *
from . import statistics
from .statistics import *
from . import matrix
from .matrix import *
from . import random
from .random import *

from . import ma
from . import fft
from . import random
from .ufunc import ufunc

from . import linalg
from .linalg.matrix_and_vector_products import (
    # dot,
    # vdot,
    inner,
    outer,
    matrix_power,
    tensordot,
    # einsum,
    # einsum_path,
    kron,
)

from .linalg.decompositions import cholesky, qr, svd

from .linalg.norms_and_other_numbers import det, slogdet, matrix_rank, norm, trace

from .linalg.solving_equations_and_inverting_matrices import pinv, inv, solve

# importing private functions for ufunc initialization #
# -----------------------------------------------------#

from ivy.functional.frontends.numpy.mathematical_functions.miscellaneous import (
    _absolute,
    _cbrt,
    _copysign,
    _fabs,
    _heaviside,
    _sign,
    _sqrt,
    _square,
)

from ivy.functional.frontends.numpy.mathematical_functions.arithmetic_operations import (
    _add,
    _divide,
    _float_power,
    _floor_divide,
    _fmod,
    _mod,
    _multiply,
    _negative,
    _positive,
    _power,
    _reciprocal,
    _subtract,
    _true_divide,
    _divmod,
)

from ivy.functional.frontends.numpy.mathematical_functions.trigonometric_functions import (
    _arccos,
    _arcsin,
    _arctan,
    _cos,
    _deg2rad,
    _rad2deg,
    _sin,
    _tan,
)

from ivy.functional.frontends.numpy.mathematical_functions.handling_complex_numbers import (
    _imag,
    _real,
)

from ivy.functional.frontends.numpy.mathematical_functions.hyperbolic_functions import (
    _arccosh,
    _arcsinh,
    _arctanh,
    _cosh,
    _sinh,
    _tanh,
)

from ivy.functional.frontends.numpy.mathematical_functions.rounding import (
    _ceil,
    _trunc,
    _floor,
)

from ivy.functional.frontends.numpy.logic.comparison import (
    _equal,
    _greater,
    _greater_equal,
    _less,
    _less_equal,
    _not_equal,
)

from ivy.functional.frontends.numpy.mathematical_functions.exponents_and_logarithms import (
    _exp,
    _exp2,
    _expm1,
    _log,
    _log10,
    _log1p,
    _log2,
    _logaddexp,
    _logaddexp2,
    _ldexp,
)

from ivy.functional.frontends.numpy.logic.array_type_testing import (
    _isfinite,
    _isinf,
    _isnan,
)

from ivy.functional.frontends.numpy.logic.logical_operations import (
    _logical_and,
    _logical_not,
    _logical_or,
    _logical_xor,
)

from ivy.functional.frontends.numpy.linalg.matrix_and_vector_products import _matmul

from ivy.functional.frontends.numpy.mathematical_functions.extrema_finding import (
    _maximum,
    _minimum,
    _fmax,
)

_frontend_array = array

# initializing ufuncs #
# ---------------------#

absolute = ufunc("_absolute")
cbrt = ufunc("_cbrt")
copysign = ufunc("_copysign")
fabs = ufunc("_fabs")
heaviside = ufunc("_heaviside")
sign = ufunc("_sign")
sqrt = ufunc("_sqrt")
square = ufunc("_square")
add = ufunc("_add")
divide = ufunc("_divide")
float_power = ufunc("_float_power")
floor_divide = ufunc("_floor_divide")
fmod = ufunc("_fmod")
mod = ufunc("_mod")
multiply = ufunc("_multiply")
negative = ufunc("_negative")
positive = ufunc("_positive")
power = ufunc("_power")
reciprocal = ufunc("_reciprocal")
subtract = ufunc("_subtract")
true_divide = ufunc("_divide")
arccos = ufunc("_arccos")
arcsin = ufunc("_arcsin")
arctan = ufunc("_arctan")
cos = ufunc("_cos")
deg2rad = ufunc("_deg2rad")
rad2deg = ufunc("_rad2deg")
sin = ufunc("_sin")
tan = ufunc("_tan")
arccosh = ufunc("_arccosh")
arcsinh = ufunc("_arcsinh")
arctanh = ufunc("_arctanh")
cosh = ufunc("_cosh")
sinh = ufunc("_sinh")
tanh = ufunc("_tanh")
ceil = ufunc("_ceil")
trunc = ufunc("_trunc")
equal = ufunc("_equal")
greater = ufunc("_greater")
greater_equal = ufunc("_greater_equal")
imag = ufunc("_imag")
less = ufunc("_less")
less_equal = ufunc("_less_equal")
not_equal = ufunc("_not_equal")
exp = ufunc("_exp")
exp2 = ufunc("_exp2")
expm1 = ufunc("_expm1")
log = ufunc("_log")
log10 = ufunc("_log10")
log1p = ufunc("_log1p")
log2 = ufunc("_log2")
logaddexp = ufunc("_logaddexp")
logaddexp2 = ufunc("_logaddexp2")
isfinite = ufunc("_isfinite")
isinf = ufunc("_isinf")
isnan = ufunc("_isnan")
logical_and = ufunc("_logical_and")
logical_not = ufunc("_logical_not")
logical_or = ufunc("_logical_or")
logical_xor = ufunc("_logical_xor")
matmul = ufunc("_matmul")
maximum = ufunc("_maximum")
minimum = ufunc("_minimum")
real = ufunc("_real")
divmod = ufunc("_divmod")
fmax = ufunc("_fmax")
ldexp = ufunc("_ldexp")
floor = ufunc("_floor")
