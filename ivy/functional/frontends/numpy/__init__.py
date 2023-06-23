# global
import sys

# local
import ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.frontends import set_frontend_to_specific_version
from typing import Union, Iterable, Tuple
from numbers import Number
from .data_type_routines import dtype
from . import ndarray
from .ndarray import *
from . import scalars
from .scalars import *


# Constructing dtypes are required as ivy.<dtype>
# will change dynamically on the backend and may not be available
_int8 = ivy.IntDtype("int8")
_int16 = ivy.IntDtype("int16")
_int32 = ivy.IntDtype("int32")
_int64 = ivy.IntDtype("int64")
_uint8 = ivy.UintDtype("uint8")
_uint16 = ivy.UintDtype("uint16")
_uint32 = ivy.UintDtype("uint32")
_uint64 = ivy.UintDtype("uint64")
_bfloat16 = ivy.FloatDtype("bfloat16")
_float16 = ivy.FloatDtype("float16")
_float32 = ivy.FloatDtype("float32")
_float64 = ivy.FloatDtype("float64")
_complex64 = ivy.ComplexDtype("complex64")
_complex128 = ivy.ComplexDtype("complex128")
_bool = ivy.Dtype("bool")

numpy_promotion_table = {
    (_bool, _bool): _bool,
    (_bool, _int8): _int8,
    (_bool, _int16): _int16,
    (_bool, _int32): _int32,
    (_bool, _int64): _int64,
    (_bool, _uint8): _uint8,
    (_bool, _uint16): _uint16,
    (_bool, _uint32): _uint32,
    (_bool, _uint64): _uint64,
    (_bool, _bfloat16): _bfloat16,
    (_bool, _float16): _float16,
    (_bool, _float32): _float32,
    (_bool, _float64): _float64,
    (_bool, _complex64): _complex64,
    (_bool, _complex128): _complex128,
    (_bool, _bool): _bool,
    (_int8, _bool): _int8,
    (_int8, _int8): _int8,
    (_int8, _int16): _int16,
    (_int8, _int32): _int32,
    (_int8, _int64): _int64,
    (_int16, _bool): _int16,
    (_int16, _int8): _int16,
    (_int16, _int16): _int16,
    (_int16, _int32): _int32,
    (_int16, _int64): _int64,
    (_int32, _bool): _int32,
    (_int32, _int8): _int32,
    (_int32, _int16): _int32,
    (_int32, _int32): _int32,
    (_int32, _int64): _int64,
    (_int64, _bool): _int64,
    (_int64, _int8): _int64,
    (_int64, _int16): _int64,
    (_int64, _int32): _int64,
    (_int64, _int64): _int64,
    (_uint8, _bool): _uint8,
    (_uint8, _uint8): _uint8,
    (_uint8, _uint16): _uint16,
    (_uint8, _uint32): _uint32,
    (_uint8, _uint64): _uint64,
    (_uint16, _bool): _uint16,
    (_uint16, _uint8): _uint16,
    (_uint16, _uint16): _uint16,
    (_uint16, _uint32): _uint32,
    (_uint16, _uint64): _uint64,
    (_uint32, _bool): _uint32,
    (_uint32, _uint8): _uint32,
    (_uint32, _uint16): _uint32,
    (_uint32, _uint32): _uint32,
    (_uint32, _uint64): _uint64,
    (_uint64, _bool): _uint64,
    (_uint64, _uint8): _uint64,
    (_uint64, _uint16): _uint64,
    (_uint64, _uint32): _uint64,
    (_uint64, _uint64): _uint64,
    (_int8, _uint8): _int16,
    (_int8, _uint16): _int32,
    (_int8, _uint32): _int64,
    (_int16, _uint8): _int16,
    (_int16, _uint16): _int32,
    (_int16, _uint32): _int64,
    (_int32, _uint8): _int32,
    (_int32, _uint16): _int32,
    (_int32, _uint32): _int64,
    (_int64, _uint8): _int64,
    (_int64, _uint16): _int64,
    (_int64, _uint32): _int64,
    (_uint8, _int8): _int16,
    (_uint16, _int8): _int32,
    (_uint32, _int8): _int64,
    (_uint8, _int16): _int16,
    (_uint16, _int16): _int32,
    (_uint32, _int16): _int64,
    (_uint8, _int32): _int32,
    (_uint16, _int32): _int32,
    (_uint32, _int32): _int64,
    (_uint8, _int64): _int64,
    (_uint16, _int64): _int64,
    (_uint32, _int64): _int64,
    (_float16, _bool): _float16,
    (_float16, _float16): _float16,
    (_float16, _float32): _float32,
    (_float16, _float64): _float64,
    (_float32, _bool): _float32,
    (_float32, _float16): _float32,
    (_float32, _float32): _float32,
    (_float32, _float64): _float64,
    (_float64, _bool): _float64,
    (_float64, _float16): _float64,
    (_float64, _float32): _float64,
    (_float64, _float64): _float64,
    (_uint64, _int8): _float64,
    (_int8, _uint64): _float64,
    (_uint64, _int16): _float64,
    (_int16, _uint64): _float64,
    (_uint64, _int32): _float64,
    (_int32, _uint64): _float64,
    (_uint64, _int64): _float64,
    (_int64, _uint64): _float64,
    (_int8, _float16): _float16,
    (_float16, _int8): _float16,
    (_int8, _float32): _float32,
    (_float32, _int8): _float32,
    (_int8, _float64): _float64,
    (_float64, _int8): _float64,
    (_int16, _float16): _float32,
    (_float16, _int16): _float32,
    (_int16, _float32): _float32,
    (_float32, _int16): _float32,
    (_int16, _float64): _float64,
    (_float64, _int16): _float64,
    (_int32, _float16): _float64,
    (_float16, _int32): _float64,
    (_int32, _float32): _float64,
    (_float32, _int32): _float64,
    (_int32, _float64): _float64,
    (_float64, _int32): _float64,
    (_int64, _float16): _float64,
    (_float16, _int64): _float64,
    (_int64, _float32): _float64,
    (_float32, _int64): _float64,
    (_int64, _float64): _float64,
    (_float64, _int64): _float64,
    (_uint8, _float16): _float16,
    (_float16, _uint8): _float16,
    (_uint8, _float32): _float32,
    (_float32, _uint8): _float32,
    (_uint8, _float64): _float64,
    (_float64, _uint8): _float64,
    (_uint16, _float16): _float32,
    (_float16, _uint16): _float32,
    (_uint16, _float32): _float32,
    (_float32, _uint16): _float32,
    (_uint16, _float64): _float64,
    (_float64, _uint16): _float64,
    (_uint32, _float16): _float64,
    (_float16, _uint32): _float64,
    (_uint32, _float32): _float64,
    (_float32, _uint32): _float64,
    (_uint32, _float64): _float64,
    (_float64, _uint32): _float64,
    (_uint64, _float16): _float64,
    (_float16, _uint64): _float64,
    (_uint64, _float32): _float64,
    (_float32, _uint64): _float64,
    (_uint64, _float64): _float64,
    (_float64, _uint64): _float64,
    (_bfloat16, _bfloat16): _bfloat16,
    (_bfloat16, _uint8): _bfloat16,
    (_uint8, _bfloat16): _bfloat16,
    (_bfloat16, _int8): _bfloat16,
    (_int8, _bfloat16): _bfloat16,
    (_bfloat16, _float32): _float32,
    (_float32, _bfloat16): _float32,
    (_bfloat16, _float64): _float64,
    (_float64, _bfloat16): _float64,
    (_complex64, _bool): _complex64,
    (_complex64, _int8): _complex64,
    (_complex64, _int16): _complex64,
    (_complex64, _int32): _complex128,
    (_complex64, _int64): _complex128,
    (_complex64, _uint8): _complex64,
    (_complex64, _uint16): _complex64,
    (_complex64, _uint32): _complex128,
    (_complex64, _uint64): _complex128,
    (_complex64, _float16): _complex64,
    (_complex64, _float32): _complex64,
    (_complex64, _float64): _complex128,
    (_complex64, _bfloat16): _complex64,
    (_complex64, _complex64): _complex64,
    (_complex64, _complex128): _complex128,
    (_complex128, _bool): _complex128,
    (_complex128, _int8): _complex128,
    (_complex128, _int16): _complex128,
    (_complex128, _int32): _complex128,
    (_complex128, _int64): _complex128,
    (_complex128, _uint8): _complex128,
    (_complex128, _uint16): _complex128,
    (_complex128, _uint32): _complex128,
    (_complex128, _uint64): _complex128,
    (_complex128, _float16): _complex128,
    (_complex128, _float32): _complex128,
    (_complex128, _float64): _complex128,
    (_complex128, _bfloat16): _complex128,
    (_complex128, _complex64): _complex128,
    (_complex128, _complex128): _complex128,
    (_int8, _complex64): _complex64,
    (_int16, _complex64): _complex64,
    (_int32, _complex64): _complex128,
    (_int64, _complex64): _complex128,
    (_uint8, _complex64): _complex64,
    (_uint16, _complex64): _complex64,
    (_uint32, _complex64): _complex128,
    (_uint64, _complex64): _complex128,
    (_float16, _complex64): _complex64,
    (_float32, _complex64): _complex64,
    (_float64, _complex64): _complex128,
    (_bfloat16, _complex64): _complex64,
    (_int8, _complex128): _complex128,
    (_int16, _complex128): _complex128,
    (_int32, _complex128): _complex128,
    (_int64, _complex128): _complex128,
    (_uint8, _complex128): _complex128,
    (_uint16, _complex128): _complex128,
    (_uint32, _complex128): _complex128,
    (_uint64, _complex128): _complex128,
    (_float16, _complex128): _complex128,
    (_float32, _complex128): _complex128,
    (_float64, _complex128): _complex128,
    (_bfloat16, _complex128): _complex128,
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
    bool_: _bool,
    number: _float64,
    integer: _int64,
    signedinteger: _int64,
    byte: _int8,
    short: _int16,
    intc: _int32,
    longlong: _int64,
    int_: _int64,
    unsignedinteger: _uint64,
    ubyte: _uint8,
    ushort: _uint16,
    uintc: _uint32,
    ulonglong: _uint64,
    uint: _uint64,
    inexact: _float64,
    floating: _float64,
    half: _float16,
    single: _float32,
    float_: _float64,
    _bfloat16: _bfloat16,
    complexfloating: _complex128,
    csingle: _complex64,
    complex_: _complex128,
}

numpy_dtype_to_scalar = {v: k for k, v in numpy_scalar_to_dtype.items()}

numpy_casting_rules = {
    _bool: [
        _bool,
        _uint8,
        _uint16,
        _uint32,
        _uint64,
        _int8,
        _int16,
        _int32,
        _int64,
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _int8: [
        _int8,
        _int16,
        _int32,
        _int64,
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _int16: [
        _int16,
        _int32,
        _int64,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _int32: [_int32, _int64, _float64, _complex128],
    _int64: [_int64, _float64, _complex128],
    _uint8: [
        _uint8,
        _uint16,
        _uint32,
        _uint64,
        _int16,
        _int32,
        _int64,
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _uint16: [
        _uint16,
        _uint32,
        _uint64,
        _int32,
        _int64,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _uint32: [
        _uint32,
        _uint64,
        _int64,
        _float64,
        _complex128,
    ],
    _uint64: [_uint64, _float64, _complex128],
    _float16: [
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _float32: [
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _float64: [_float64, _complex128],
    _complex64: [_complex64, _complex128],
    _complex128: [_complex128],
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
    Promote the dtype of the given ivy array inputs to a common dtype based on numpy
    type promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
    """
    # ToDo: Overflows not working properly for numpy, if a scalar or 0-dim
    #   is passed with an array, it should go to the next largest dtype that
    #   can hold the value without overflow. E.g a np.array([5], 'int8') + 300 operation
    #   results in np.array([305]) with int16 dtype
    x1 = ivy.asarray(x1)
    x2 = ivy.asarray(x2)
    type1 = ivy.default_dtype(item=x1).strip("u123456789")
    type2 = ivy.default_dtype(item=x2).strip("u123456789")
    # Ignore type of 0-dim arrays or scalars to mimic numpy
    if not x1.shape == () and x2.shape == () and type1 == type2:
        x2 = ivy.asarray(
            x2, dtype=x1.dtype, device=ivy.default_device(item=x1, as_native=False)
        )
    elif x1.shape == () and not x2.shape == () and type1 == type2:
        x1 = ivy.asarray(
            x1, dtype=x2.dtype, device=ivy.default_device(item=x2, as_native=False)
        )
    else:
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
    cross,
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
    _lcm,
    _gcd,
    _clip,
)

from ivy.functional.frontends.numpy.mathematical_functions.arithmetic_operations import (  # noqa
    _add,
    _divide,
    _float_power,
    _floor_divide,
    _fmod,
    _mod,
    _modf,
    _multiply,
    _negative,
    _positive,
    _power,
    _reciprocal,
    _subtract,
    _divmod,
)

from ivy.functional.frontends.numpy.mathematical_functions.trigonometric_functions import (  # noqa
    _arccos,
    _arcsin,
    _arctan,
    _cos,
    _deg2rad,
    _rad2deg,
    _sin,
    _tan,
    _degrees,
)

from ivy.functional.frontends.numpy.mathematical_functions.handling_complex_numbers import (  # noqa
    _conj,
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
    _rint,
)

from ivy.functional.frontends.numpy.logic.comparison import (
    _equal,
    _greater,
    _greater_equal,
    _less,
    _less_equal,
    _not_equal,
)

from ivy.functional.frontends.numpy.mathematical_functions.exponents_and_logarithms import (  # noqa
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
    _frexp,
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
    _fmin,
)

from ivy.functional.frontends.numpy.mathematical_functions.floating_point_routines import (  # noqa
    _nextafter,
    _spacing,
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
modf = ufunc("_modf")
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
degrees = ufunc("_degrees")
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
divmod = ufunc("_divmod")
fmax = ufunc("_fmax")
fmin = ufunc("_fmin")
ldexp = ufunc("_ldexp")
floor = ufunc("_floor")
frexp = ufunc("_frexp")
conj = ufunc("_conj")
rint = ufunc("_rint")
nextafter = ufunc("_nextafter")
conjugate = ufunc("_conj")
lcm = ufunc("_lcm")
gcd = ufunc("_gcd")
spacing = ufunc("_spacing")
clip = ufunc("_clip")

# setting to specific version #
# --------------------------- #

set_frontend_to_specific_version(sys.modules[__name__])
