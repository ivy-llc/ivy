# flake8: noqa
# global
from numbers import Number
from typing import Union, Tuple, Iterable


# local
import ivy
from ivy.exceptions import handle_exceptions
from ivy.functional.frontends.numpy import dtype


int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
bfloat16 = dtype("bfloat16")
float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
complex64 = dtype("complex64")
complex128 = dtype("complex128")
complex256 = dtype("complex256")
bool = dtype("bool")


# jax-numpy casting table
jax_numpy_casting_table = {
    ivy.bool: [
        ivy.bool,
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
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
        ivy.bfloat16,
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
    ivy.int32: [
        ivy.int32,
        ivy.int64,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.int64: [
        ivy.int64,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint8: [
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.uint16: [
        ivy.int32,
        ivy.int64,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint32: [
        ivy.int64,
        ivy.uint32,
        ivy.uint64,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.uint64: [
        ivy.uint64,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
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
    ivy.float64: [
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.complex64: [ivy.complex64, ivy.complex128, ivy.complex256, ivy.bfloat16],
    ivy.complex128: [ivy.complex128, ivy.complex256, ivy.bfloat16],
    ivy.complex256: [ivy.complex256, ivy.bfloat16],
    ivy.bfloat16: [
        ivy.bfloat16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
}


# jax-numpy type promotion table
# data type promotion
jax_promotion_table = {
    (ivy.bool, ivy.bool): ivy.bool,
    (ivy.bool, ivy.uint8): ivy.uint8,
    (ivy.bool, ivy.uint16): ivy.uint16,
    (ivy.bool, ivy.uint32): ivy.uint32,
    (ivy.bool, ivy.uint64): ivy.uint64,
    (ivy.bool, ivy.int8): ivy.int8,
    (ivy.bool, ivy.int16): ivy.int16,
    (ivy.bool, ivy.int32): ivy.int32,
    (ivy.bool, ivy.int64): ivy.int64,
    (ivy.bool, ivy.bfloat16): ivy.bfloat16,
    (ivy.bool, ivy.float16): ivy.float16,
    (ivy.bool, ivy.float32): ivy.float32,
    (ivy.bool, ivy.float64): ivy.float64,
    (ivy.uint8, ivy.bool): ivy.uint8,
    (ivy.uint8, ivy.uint8): ivy.uint8,
    (ivy.uint8, ivy.uint16): ivy.uint16,
    (ivy.uint8, ivy.uint32): ivy.uint32,
    (ivy.uint8, ivy.uint64): ivy.uint64,
    (ivy.uint8, ivy.int8): ivy.int16,
    (ivy.uint8, ivy.int16): ivy.int16,
    (ivy.uint8, ivy.int32): ivy.int32,
    (ivy.uint8, ivy.int64): ivy.int64,
    (ivy.uint8, ivy.bfloat16): ivy.bfloat16,
    (ivy.uint8, ivy.float16): ivy.float16,
    (ivy.uint8, ivy.float32): ivy.float32,
    (ivy.uint8, ivy.float64): ivy.float64,
    (ivy.uint16, ivy.bool): ivy.uint16,
    (ivy.uint16, ivy.uint8): ivy.uint16,
    (ivy.uint16, ivy.uint16): ivy.uint16,
    (ivy.uint16, ivy.uint32): ivy.uint32,
    (ivy.uint16, ivy.uint64): ivy.uint64,
    (ivy.uint16, ivy.int8): ivy.int32,
    (ivy.uint16, ivy.int16): ivy.int32,
    (ivy.uint16, ivy.int32): ivy.int32,
    (ivy.uint16, ivy.int64): ivy.int64,
    (ivy.uint16, ivy.bfloat16): ivy.bfloat16,
    (ivy.uint16, ivy.float16): ivy.float16,
    (ivy.uint16, ivy.float32): ivy.float32,
    (ivy.uint16, ivy.float64): ivy.float64,
    (ivy.uint32, ivy.bool): ivy.uint32,
    (ivy.uint32, ivy.uint8): ivy.uint32,
    (ivy.uint32, ivy.uint16): ivy.uint32,
    (ivy.uint32, ivy.uint32): ivy.uint32,
    (ivy.uint32, ivy.uint64): ivy.uint64,
    (ivy.uint32, ivy.int8): ivy.int64,
    (ivy.uint32, ivy.int16): ivy.int64,
    (ivy.uint32, ivy.int32): ivy.int64,
    (ivy.uint32, ivy.int64): ivy.int64,
    (ivy.uint32, ivy.bfloat16): ivy.bfloat16,
    (ivy.uint32, ivy.float16): ivy.float16,
    (ivy.uint32, ivy.float32): ivy.float32,
    (ivy.uint32, ivy.float64): ivy.float64,
    (ivy.uint64, ivy.bool): ivy.uint64,
    (ivy.uint64, ivy.uint8): ivy.uint64,
    (ivy.uint64, ivy.uint16): ivy.uint64,
    (ivy.uint64, ivy.uint32): ivy.uint64,
    (ivy.uint64, ivy.uint64): ivy.uint64,
    (ivy.uint64, ivy.int8): float,
    (ivy.uint64, ivy.int16): float,
    (ivy.uint64, ivy.int32): float,
    (ivy.uint64, ivy.int64): float,
    (ivy.uint64, ivy.bfloat16): ivy.bfloat16,
    (ivy.uint64, ivy.float16): ivy.float16,
    (ivy.uint64, ivy.float32): ivy.float32,
    (ivy.uint64, ivy.float64): ivy.float64,
    (ivy.int8, ivy.bool): ivy.int8,
    (ivy.int8, ivy.uint8): ivy.int16,
    (ivy.int8, ivy.uint16): ivy.int32,
    (ivy.int8, ivy.uint32): ivy.int64,
    (ivy.int8, ivy.uint64): float,
    (ivy.int8, ivy.int8): ivy.int8,
    (ivy.int8, ivy.int16): ivy.int16,
    (ivy.int8, ivy.int32): ivy.int32,
    (ivy.int8, ivy.int64): ivy.int64,
    (ivy.int8, ivy.bfloat16): ivy.bfloat16,
    (ivy.int8, ivy.float16): ivy.float16,
    (ivy.int8, ivy.float32): ivy.float32,
    (ivy.int8, ivy.float64): ivy.float64,
    (ivy.int16, ivy.bool): ivy.int16,
    (ivy.int16, ivy.uint8): ivy.int16,
    (ivy.int16, ivy.uint16): ivy.int32,
    (ivy.int16, ivy.uint32): ivy.int64,
    (ivy.int16, ivy.uint64): float,
    (ivy.int16, ivy.int8): ivy.int16,
    (ivy.int16, ivy.int16): ivy.int16,
    (ivy.int16, ivy.int32): ivy.int32,
    (ivy.int16, ivy.int64): ivy.int64,
    (ivy.int16, ivy.bfloat16): ivy.bfloat16,
    (ivy.int16, ivy.float16): ivy.float16,
    (ivy.int16, ivy.float32): ivy.float32,
    (ivy.int16, ivy.float64): ivy.float64,
    (ivy.int32, ivy.bool): ivy.int32,
    (ivy.int32, ivy.uint8): ivy.int32,
    (ivy.int32, ivy.uint16): ivy.int32,
    (ivy.int32, ivy.uint32): ivy.int64,
    (ivy.int32, ivy.uint64): float,
    (ivy.int32, ivy.int8): ivy.int32,
    (ivy.int32, ivy.int16): ivy.int32,
    (ivy.int32, ivy.int32): ivy.int32,
    (ivy.int32, ivy.int64): ivy.int64,
    (ivy.int32, ivy.bfloat16): ivy.bfloat16,
    (ivy.int32, ivy.float16): ivy.float16,
    (ivy.int32, ivy.float32): ivy.float32,
    (ivy.int32, ivy.float64): ivy.float64,
    (ivy.int64, ivy.bool): ivy.int64,
    (ivy.int64, ivy.uint8): ivy.int64,
    (ivy.int64, ivy.uint16): ivy.int64,
    (ivy.int64, ivy.uint32): ivy.int64,
    (ivy.int64, ivy.uint64): float,
    (ivy.int64, ivy.int8): ivy.int64,
    (ivy.int64, ivy.int16): ivy.int64,
    (ivy.int64, ivy.int32): ivy.int64,
    (ivy.int64, ivy.int64): ivy.int64,
    (ivy.int64, ivy.bfloat16): ivy.bfloat16,
    (ivy.int64, ivy.float16): ivy.float16,
    (ivy.int64, ivy.float32): ivy.float32,
    (ivy.int64, ivy.float64): ivy.float64,
    (ivy.bfloat16, ivy.bool): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint8): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint32): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint64): ivy.bfloat16,
    (ivy.bfloat16, ivy.int8): ivy.bfloat16,
    (ivy.bfloat16, ivy.int16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int32): ivy.bfloat16,
    (ivy.bfloat16, ivy.int64): ivy.bfloat16,
    (ivy.bfloat16, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.float16): ivy.float32,
    (ivy.bfloat16, ivy.float32): ivy.float32,
    (ivy.bfloat16, ivy.float64): ivy.float64,
    (ivy.float16, ivy.bool): ivy.float16,
    (ivy.float16, ivy.uint8): ivy.float16,
    (ivy.float16, ivy.uint16): ivy.float16,
    (ivy.float16, ivy.uint32): ivy.float16,
    (ivy.float16, ivy.uint64): ivy.float16,
    (ivy.float16, ivy.int8): ivy.float16,
    (ivy.float16, ivy.int16): ivy.float16,
    (ivy.float16, ivy.int32): ivy.float16,
    (ivy.float16, ivy.int64): ivy.float16,
    (ivy.float16, ivy.bfloat16): ivy.float64,
    (ivy.float16, ivy.float16): ivy.float16,
    (ivy.float16, ivy.float32): ivy.float32,
    (ivy.float16, ivy.float64): ivy.float64,
    (ivy.float32, ivy.bool): ivy.float32,
    (ivy.float32, ivy.uint8): ivy.float32,
    (ivy.float32, ivy.uint16): ivy.float32,
    (ivy.float32, ivy.uint32): ivy.float32,
    (ivy.float32, ivy.uint64): ivy.float32,
    (ivy.float32, ivy.int8): ivy.float32,
    (ivy.float32, ivy.int16): ivy.float32,
    (ivy.float32, ivy.int32): ivy.float32,
    (ivy.float32, ivy.int64): ivy.float32,
    (ivy.float32, ivy.bfloat16): ivy.float32,
    (ivy.float32, ivy.float16): ivy.float32,
    (ivy.float32, ivy.float32): ivy.float32,
    (ivy.float32, ivy.float64): ivy.float64,
    (ivy.float64, ivy.bool): ivy.float64,
    (ivy.float64, ivy.uint8): ivy.float64,
    (ivy.float64, ivy.uint16): ivy.float64,
    (ivy.float64, ivy.uint32): ivy.float64,
    (ivy.float64, ivy.uint64): ivy.float64,
    (ivy.float64, ivy.int8): ivy.float64,
    (ivy.float64, ivy.int16): ivy.float64,
    (ivy.float64, ivy.int32): ivy.float64,
    (ivy.float64, ivy.int64): ivy.float64,
    (ivy.float64, ivy.bfloat16): ivy.float64,
    (ivy.float64, ivy.float16): ivy.float64,
    (ivy.float64, ivy.float32): ivy.float64,
    (ivy.float64, ivy.float64): ivy.float64,
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
from .dtype import can_cast, promote_types

from .._src.numpy.lax_numpy import _rewriting_take
