# flake8: noqa
import ivy
from ivy.exceptions import handle_exceptions
from typing import Union, Iterable, Tuple
from numbers import Number
from .data_type_routines import dtype
from . import ndarray
from .ndarray import *
from . import scalars
from .scalars import *


numpy_promotion_table = {
    (ivy.bool, ivy.bool): ivy.bool,
    (ivy.bool, ivy.int8): ivy.int8,
    (ivy.bool, ivy.int16): ivy.int16,
    (ivy.bool, ivy.int32): ivy.int32,
    (ivy.bool, ivy.int64): ivy.int64,
    (ivy.bool, ivy.uint8): ivy.uint8,
    (ivy.bool, ivy.uint16): ivy.uint16,
    (ivy.bool, ivy.uint32): ivy.uint32,
    (ivy.bool, ivy.uint64): ivy.uint64,
    (ivy.bool, ivy.bfloat16): ivy.bfloat16,
    (ivy.bool, ivy.float16): ivy.float16,
    (ivy.bool, ivy.float32): ivy.float32,
    (ivy.bool, ivy.float64): ivy.float64,
    (ivy.bool, ivy.complex64): ivy.complex64,
    (ivy.bool, ivy.complex128): ivy.complex128,
    (ivy.bool, ivy.complex256): ivy.complex256,
    (ivy.bool, ivy.bool): ivy.bool,
    (ivy.int8, ivy.bool): ivy.int8,
    (ivy.int8, ivy.int8): ivy.int8,
    (ivy.int8, ivy.int16): ivy.int16,
    (ivy.int8, ivy.int32): ivy.int32,
    (ivy.int8, ivy.int64): ivy.int64,
    (ivy.int16, ivy.bool): ivy.int16,
    (ivy.int16, ivy.int8): ivy.int16,
    (ivy.int16, ivy.int16): ivy.int16,
    (ivy.int16, ivy.int32): ivy.int32,
    (ivy.int16, ivy.int64): ivy.int64,
    (ivy.int32, ivy.bool): ivy.int32,
    (ivy.int32, ivy.int8): ivy.int32,
    (ivy.int32, ivy.int16): ivy.int32,
    (ivy.int32, ivy.int32): ivy.int32,
    (ivy.int32, ivy.int64): ivy.int64,
    (ivy.int64, ivy.bool): ivy.int64,
    (ivy.int64, ivy.int8): ivy.int64,
    (ivy.int64, ivy.int16): ivy.int64,
    (ivy.int64, ivy.int32): ivy.int64,
    (ivy.int64, ivy.int64): ivy.int64,
    (ivy.uint8, ivy.bool): ivy.uint8,
    (ivy.uint8, ivy.uint8): ivy.uint8,
    (ivy.uint8, ivy.uint16): ivy.uint16,
    (ivy.uint8, ivy.uint32): ivy.uint32,
    (ivy.uint8, ivy.uint64): ivy.uint64,
    (ivy.uint16, ivy.bool): ivy.uint16,
    (ivy.uint16, ivy.uint8): ivy.uint16,
    (ivy.uint16, ivy.uint16): ivy.uint16,
    (ivy.uint16, ivy.uint32): ivy.uint32,
    (ivy.uint16, ivy.uint64): ivy.uint64,
    (ivy.uint32, ivy.bool): ivy.uint32,
    (ivy.uint32, ivy.uint8): ivy.uint32,
    (ivy.uint32, ivy.uint16): ivy.uint32,
    (ivy.uint32, ivy.uint32): ivy.uint32,
    (ivy.uint32, ivy.uint64): ivy.uint64,
    (ivy.uint64, ivy.bool): ivy.uint64,
    (ivy.uint64, ivy.uint8): ivy.uint64,
    (ivy.uint64, ivy.uint16): ivy.uint64,
    (ivy.uint64, ivy.uint32): ivy.uint64,
    (ivy.uint64, ivy.uint64): ivy.uint64,
    (ivy.int8, ivy.uint8): ivy.int16,
    (ivy.int8, ivy.uint16): ivy.int32,
    (ivy.int8, ivy.uint32): ivy.int64,
    (ivy.int16, ivy.uint8): ivy.int16,
    (ivy.int16, ivy.uint16): ivy.int32,
    (ivy.int16, ivy.uint32): ivy.int64,
    (ivy.int32, ivy.uint8): ivy.int32,
    (ivy.int32, ivy.uint16): ivy.int32,
    (ivy.int32, ivy.uint32): ivy.int64,
    (ivy.int64, ivy.uint8): ivy.int64,
    (ivy.int64, ivy.uint16): ivy.int64,
    (ivy.int64, ivy.uint32): ivy.int64,
    (ivy.uint8, ivy.int8): ivy.int16,
    (ivy.uint16, ivy.int8): ivy.int32,
    (ivy.uint32, ivy.int8): ivy.int64,
    (ivy.uint8, ivy.int16): ivy.int16,
    (ivy.uint16, ivy.int16): ivy.int32,
    (ivy.uint32, ivy.int16): ivy.int64,
    (ivy.uint8, ivy.int32): ivy.int32,
    (ivy.uint16, ivy.int32): ivy.int32,
    (ivy.uint32, ivy.int32): ivy.int64,
    (ivy.uint8, ivy.int64): ivy.int64,
    (ivy.uint16, ivy.int64): ivy.int64,
    (ivy.uint32, ivy.int64): ivy.int64,
    (ivy.float16, ivy.bool): ivy.float16,
    (ivy.float16, ivy.float16): ivy.float16,
    (ivy.float16, ivy.float32): ivy.float32,
    (ivy.float16, ivy.float64): ivy.float64,
    (ivy.float32, ivy.bool): ivy.float32,
    (ivy.float32, ivy.float16): ivy.float32,
    (ivy.float32, ivy.float32): ivy.float32,
    (ivy.float32, ivy.float64): ivy.float64,
    (ivy.float64, ivy.bool): ivy.float64,
    (ivy.float64, ivy.float16): ivy.float64,
    (ivy.float64, ivy.float32): ivy.float64,
    (ivy.float64, ivy.float64): ivy.float64,
    (ivy.uint64, ivy.int8): ivy.float64,
    (ivy.int8, ivy.uint64): ivy.float64,
    (ivy.uint64, ivy.int16): ivy.float64,
    (ivy.int16, ivy.uint64): ivy.float64,
    (ivy.uint64, ivy.int32): ivy.float64,
    (ivy.int32, ivy.uint64): ivy.float64,
    (ivy.uint64, ivy.int64): ivy.float64,
    (ivy.int64, ivy.uint64): ivy.float64,
    (ivy.int8, ivy.float16): ivy.float16,
    (ivy.float16, ivy.int8): ivy.float16,
    (ivy.int8, ivy.float32): ivy.float32,
    (ivy.float32, ivy.int8): ivy.float32,
    (ivy.int8, ivy.float64): ivy.float64,
    (ivy.float64, ivy.int8): ivy.float64,
    (ivy.int16, ivy.float16): ivy.float32,
    (ivy.float16, ivy.int16): ivy.float32,
    (ivy.int16, ivy.float32): ivy.float32,
    (ivy.float32, ivy.int16): ivy.float32,
    (ivy.int16, ivy.float64): ivy.float64,
    (ivy.float64, ivy.int16): ivy.float64,
    (ivy.int32, ivy.float16): ivy.float64,
    (ivy.float16, ivy.int32): ivy.float64,
    (ivy.int32, ivy.float32): ivy.float64,
    (ivy.float32, ivy.int32): ivy.float64,
    (ivy.int32, ivy.float64): ivy.float64,
    (ivy.float64, ivy.int32): ivy.float64,
    (ivy.int64, ivy.float16): ivy.float64,
    (ivy.float16, ivy.int64): ivy.float64,
    (ivy.int64, ivy.float32): ivy.float64,
    (ivy.float32, ivy.int64): ivy.float64,
    (ivy.int64, ivy.float64): ivy.float64,
    (ivy.float64, ivy.int64): ivy.float64,
    (ivy.uint8, ivy.float16): ivy.float16,
    (ivy.float16, ivy.uint8): ivy.float16,
    (ivy.uint8, ivy.float32): ivy.float32,
    (ivy.float32, ivy.uint8): ivy.float32,
    (ivy.uint8, ivy.float64): ivy.float64,
    (ivy.float64, ivy.uint8): ivy.float64,
    (ivy.uint16, ivy.float16): ivy.float32,
    (ivy.float16, ivy.uint16): ivy.float32,
    (ivy.uint16, ivy.float32): ivy.float32,
    (ivy.float32, ivy.uint16): ivy.float32,
    (ivy.uint16, ivy.float64): ivy.float64,
    (ivy.float64, ivy.uint16): ivy.float64,
    (ivy.uint32, ivy.float16): ivy.float64,
    (ivy.float16, ivy.uint32): ivy.float64,
    (ivy.uint32, ivy.float32): ivy.float64,
    (ivy.float32, ivy.uint32): ivy.float64,
    (ivy.uint32, ivy.float64): ivy.float64,
    (ivy.float64, ivy.uint32): ivy.float64,
    (ivy.uint64, ivy.float16): ivy.float64,
    (ivy.float16, ivy.uint64): ivy.float64,
    (ivy.uint64, ivy.float32): ivy.float64,
    (ivy.float32, ivy.uint64): ivy.float64,
    (ivy.uint64, ivy.float64): ivy.float64,
    (ivy.float64, ivy.uint64): ivy.float64,
    (ivy.bfloat16, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint8): ivy.bfloat16,
    (ivy.uint8, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int8): ivy.bfloat16,
    (ivy.int8, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.float32): ivy.float32,
    (ivy.float32, ivy.bfloat16): ivy.float32,
    (ivy.bfloat16, ivy.float64): ivy.float64,
    (ivy.float64, ivy.bfloat16): ivy.float64,
    (ivy.complex64, ivy.bool): ivy.complex64,
    (ivy.complex64, ivy.complex64): ivy.complex64,
    (ivy.complex64, ivy.complex128): ivy.complex128,
    (ivy.complex64, ivy.complex256): ivy.complex256,
    (ivy.complex128, ivy.bool): ivy.complex128,
    (ivy.complex128, ivy.complex64): ivy.complex128,
    (ivy.complex128, ivy.complex128): ivy.complex128,
    (ivy.complex128, ivy.complex256): ivy.complex256,
    (ivy.complex256, ivy.bool): ivy.complex256,
    (ivy.complex256, ivy.complex64): ivy.complex256,
    (ivy.complex256, ivy.complex128): ivy.complex256,
    (ivy.complex256, ivy.complex256): ivy.complex256,
}

numpy_str_to_type_table = {
    "b": "int8",
    "h": "int16",
    "i": "int32",
    "l": "int64",
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
    "G": "complex256",
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
    "complex256": ("G", 16),
}

numpy_scalar_to_dtype = {
    bool_: ivy.bool,
    number: ivy.float64,
    integer: ivy.int64,
    signedinteger: ivy.int64,
    byte: ivy.int8,
    short: ivy.int16,
    intc: ivy.int32,
    longlong: ivy.int64,
    int_: ivy.int64,
    unsignedinteger: ivy.uint64,
    ubyte: ivy.uint8,
    ushort: ivy.uint16,
    uintc: ivy.uint32,
    ulonglong: ivy.uint64,
    uint: ivy.uint64,
    inexact: ivy.float64,
    floating: ivy.float64,
    half: ivy.float16,
    single: ivy.float32,
    float_: ivy.float64,
    complexfloating: ivy.complex128,
    csingle: ivy.complex64,
    complex_: ivy.complex128,
    clongfloat: ivy.complex256,
}

numpy_dtype_to_scalar = {v: k for k, v in numpy_scalar_to_dtype.items()}

numpy_casting_rules = {
    ivy.bool: [
        ivy.bool,
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.int8: [
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.int16: [
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.int32: [ivy.int32, ivy.int64, ivy.float64, ivy.complex128, ivy.complex256],
    ivy.int64: [ivy.int64, ivy.float64, ivy.complex128, ivy.complex256],
    ivy.uint8: [
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint16: [
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.int32,
        ivy.int64,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint32: [
        ivy.uint32,
        ivy.uint64,
        ivy.int64,
        ivy.float64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint64: [ivy.uint64, ivy.float64, ivy.complex128, ivy.complex256],
    ivy.float16: [
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.float32: [
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.float64: [ivy.float64, ivy.complex128, ivy.complex256],
    ivy.complex64: [ivy.complex64, ivy.complex128, ivy.complex256],
    ivy.complex128: [ivy.complex128, ivy.complex256],
    ivy.complex256: [ivy.complex256],
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
        raise ivy.exceptions.IvyException("these dtypes are not type promotable")


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
    type1 = ivy.default_dtype(item=x1).strip("u123456789")
    type2 = ivy.default_dtype(item=x2).strip("u123456789")
    if hasattr(x1, "dtype") and not hasattr(x2, "dtype") and type1 == type2:
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2, dtype=x1.dtype)
    elif not hasattr(x1, "dtype") and hasattr(x2, "dtype") and type1 == type2:
        x1 = ivy.asarray(x1, dtype=x2.dtype)
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
from . import ndarray
from . import ufunc

from . import linalg
from .linalg.matrix_and_vector_products import (
    # dot,
    # vdot,
    inner,
    outer,
    matmul,
    matrix_power,
    tensordot,
    # einsum,
    # einsum_path,
    # kron,
)

from .linalg.decompositions import cholesky, qr, svd

from .linalg.norms_and_other_numbers import det, slogdet, matrix_rank, norm, trace

from .linalg.solving_equations_and_inverting_matrices import pinv, inv, solve
