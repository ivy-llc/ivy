# flake8: noqa
# global
from numbers import Number
from typing import Union, Tuple, Iterable


# local
import ivy
from ivy.exceptions import handle_exceptions
from ivy import (
    bool,
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
    complex64,
    complex128,
    complex256,
    bfloat16,
)


# jax-numpy casting table
jax_numpy_casting_table = {
    bool: [
        bool,
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
        complex64,
        complex128,
        complex256,
        bfloat16,
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
        complex256,
        bfloat16,
    ],
    int16: [
        int16,
        int32,
        int64,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    int32: [
        int32,
        int64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    int64: [
        int64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint8: [
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
        complex64,
        complex128,
        complex256,
        bfloat16,
    ],
    uint16: [
        int32,
        int64,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint32: [
        int64,
        uint32,
        uint64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint64: [
        uint64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float16: [
        float16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float32: [
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float64: [
        float64,
        complex64,
        complex128,
        complex256,
    ],
    complex64: [complex64, complex128, complex256, bfloat16],
    complex128: [complex128, complex256, bfloat16],
    complex256: [complex256, bfloat16],
    bfloat16: [
        bfloat16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
}


# jax-numpy type promotion table
# data type promotion
jax_promotion_table = {
    (bool, bool): bool,
    (bool, uint8): uint8,
    (bool, uint16): uint16,
    (bool, uint32): uint32,
    (bool, uint64): uint64,
    (bool, int8): int8,
    (bool, int16): int16,
    (bool, int32): int32,
    (bool, int64): int64,
    (bool, bfloat16): bfloat16,
    (bool, float16): float16,
    (bool, float32): float32,
    (bool, float64): float64,
    (uint8, bool): uint8,
    (uint8, uint8): uint8,
    (uint8, uint16): uint16,
    (uint8, uint32): uint32,
    (uint8, uint64): uint64,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint8, bfloat16): bfloat16,
    (uint8, float16): float16,
    (uint8, float32): float32,
    (uint8, float64): float64,
    (uint16, bool): uint16,
    (uint16, uint8): uint16,
    (uint16, uint16): uint16,
    (uint16, uint32): uint32,
    (uint16, uint64): uint64,
    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,
    (uint16, bfloat16): bfloat16,
    (uint16, float16): float16,
    (uint16, float32): float32,
    (uint16, float64): float64,
    (uint32, bool): uint32,
    (uint32, uint8): uint32,
    (uint32, uint16): uint32,
    (uint32, uint32): uint32,
    (uint32, uint64): uint64,
    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,
    (uint32, bfloat16): bfloat16,
    (uint32, float16): float16,
    (uint32, float32): float32,
    (uint32, float64): float64,
    (uint64, bool): uint64,
    (uint64, uint8): uint64,
    (uint64, uint16): uint64,
    (uint64, uint32): uint64,
    (uint64, uint64): uint64,
    (uint64, int8): float,
    (uint64, int16): float,
    (uint64, int32): float,
    (uint64, int64): float,
    (uint64, bfloat16): bfloat16,
    (uint64, float16): float16,
    (uint64, float32): float32,
    (uint64, float64): float64,
    (int8, bool): int8,
    (int8, uint8): int16,
    (int8, uint16): int32,
    (int8, uint32): int64,
    (int8, uint64): float,
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int8, bfloat16): bfloat16,
    (int8, float16): float16,
    (int8, float32): float32,
    (int8, float64): float64,
    (int16, bool): int16,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int16, uint64): float,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int16, bfloat16): bfloat16,
    (int16, float16): float16,
    (int16, float32): float32,
    (int16, float64): float64,
    (int32, bool): int32,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int32, uint64): float,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int32, bfloat16): bfloat16,
    (int32, float16): float16,
    (int32, float32): float32,
    (int32, float64): float64,
    (int64, bool): int64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    (int64, uint64): float,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (int64, bfloat16): bfloat16,
    (int64, float16): float16,
    (int64, float32): float32,
    (int64, float64): float64,
    (bfloat16, bool): bfloat16,
    (bfloat16, uint8): bfloat16,
    (bfloat16, uint16): bfloat16,
    (bfloat16, uint32): bfloat16,
    (bfloat16, uint64): bfloat16,
    (bfloat16, int8): bfloat16,
    (bfloat16, int16): bfloat16,
    (bfloat16, int32): bfloat16,
    (bfloat16, int64): bfloat16,
    (bfloat16, bfloat16): bfloat16,
    (bfloat16, float16): float32,
    (bfloat16, float32): float32,
    (bfloat16, float64): float64,
    (float16, bool): float16,
    (float16, uint8): float16,
    (float16, uint16): float16,
    (float16, uint32): float16,
    (float16, uint64): float16,
    (float16, int8): float16,
    (float16, int16): float16,
    (float16, int32): float16,
    (float16, int64): float16,
    (float16, bfloat16): float64,
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float32, bool): float32,
    (float32, uint8): float32,
    (float32, uint16): float32,
    (float32, uint32): float32,
    (float32, uint64): float32,
    (float32, int8): float32,
    (float32, int16): float32,
    (float32, int32): float32,
    (float32, int64): float32,
    (float32, bfloat16): float32,
    (float32, float16): float32,
    (float32, float32): float32,
    (float32, float64): float64,
    (float64, bool): float64,
    (float64, uint8): float64,
    (float64, uint16): float64,
    (float64, uint32): float64,
    (float64, uint64): float64,
    (float64, int8): float64,
    (float64, int16): float64,
    (float64, int32): float64,
    (float64, int64): float64,
    (float64, bfloat16): float64,
    (float64, float16): float64,
    (float64, float32): float64,
    (float64, float64): float64,
}


@handle_exceptions
def promote_types_jax(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
) -> ivy.Dtype:
    """
    Promotes the datatypes type1 and type2, returning the data type they promote to
    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote
    Returns
    -------
    ret
        The type that both input types promote to
    """
    try:
        ret = jax_promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]
    except KeyError:
        raise ivy.exceptions.IvyException("these dtypes are not type promotable")
    return ret


@handle_exceptions
def promote_types_of_jax_inputs(
    x1: Union[ivy.Array, Number, Iterable[Number]],
    x2: Union[ivy.Array, Number, Iterable[Number]],
    /,
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Promotes the dtype of the given native array inputs to a common dtype
    based on type promotion rules. While passing float or integer values or any
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
        if x1.dtype != x2.dtype:
            promoted = promote_types_jax(x1.dtype, x2.dtype)
            x1 = ivy.asarray(x1, dtype=promoted)
            x2 = ivy.asarray(x2, dtype=promoted)
    elif hasattr(x1, "dtype"):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2, dtype=x1.dtype)
    else:
        x1 = ivy.asarray(x1, dtype=x2.dtype)
        x2 = ivy.asarray(x2)
    return x1, x2


from . import fft
from . import linalg
from . import creation
from .creation import *
from . import name_space_functions
from .name_space_functions import *
from . import dtype
from .dtype import *

from .._src.numpy.lax_numpy import _rewriting_take
