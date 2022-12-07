# flake8: noqa
import ivy
from ivy.exceptions import handle_exceptions
from typing import Union, Iterable, Tuple
from numbers import Number

from ivy import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    bool,
    bfloat16,
    complex64,
    complex128,
    complex256,
)

numpy_promotion_table = {
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
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex64, complex256): complex256,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
    (complex128, complex256): complex256,
    (complex256, complex64): complex256,
    (complex256, complex128): complex256,
    (complex256, complex256): complex256,
}

numpy_str_to_type_table = {
    "b": int8,
    "h": int16,
    "i": int32,
    "l": int64,
    "B": uint8,
    "H": uint16,
    "I": uint32,
    "L": uint64,
    "e": float16,
    "f": float32,
    "d": float64,
    "?": bool,
    "E": bfloat16,
    "F": complex64,
    "D": complex128,
    "G": complex256,
}

numpy_type_to_str_and_num_table = {
    int8: ("b", 1),
    int16: ("h", 3),
    int32: ("i", 5),
    int64: ("l", 7),
    uint8: ("B", 2),
    uint16: ("H", 4),
    uint32: ("I", 6),
    uint64: ("L", 8),
    float16: ("e", 23),
    float32: ("f", 11),
    float64: ("d", 12),
    bool: ("?", 0),
    bfloat16: ("E", 256),
    complex64: ("F", 14),
    complex128: ("D", 15),
    complex256: ("G", 16),
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
    if (hasattr(x1, "dtype") and hasattr(x2, "dtype")) or (
        not hasattr(x1, "dtype") and not hasattr(x2, "dtype")
    ):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2)
        promoted = promote_numpy_dtypes(x1.dtype, x2.dtype)
        x1 = ivy.asarray(x1, dtype=promoted)
        x2 = ivy.asarray(x2, dtype=promoted)
    elif hasattr(x1, "dtype"):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2, dtype=x1.dtype)
    else:
        x1 = ivy.asarray(x1, dtype=x2.dtype)
        x2 = ivy.asarray(x2)
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
from . import ndarray
from .ndarray import *
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
