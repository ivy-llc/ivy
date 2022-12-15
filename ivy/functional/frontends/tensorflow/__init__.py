# flake8: noqa
# local
from ivy.exceptions import handle_exceptions
import ivy
from numbers import Number
from typing import Union, Tuple, Iterable
from .dtypes import DType

tensorflow_enum_to_type = {
    1: ivy.float32,
    2: ivy.float64,
    3: ivy.int32,
    4: ivy.uint8,
    5: ivy.int16,
    6: ivy.int8,
    8: ivy.complex64,
    9: ivy.int64,
    10: ivy.bool,
    14: ivy.bfloat16,
    17: ivy.uint16,
    18: ivy.complex128,
    19: ivy.float16,
    22: ivy.uint32,
    23: ivy.uint64,
}

tensorflow_type_to_enum = {v: k for k, v in tensorflow_enum_to_type.items()}


float32 = DType(1)
float64 = DType(2)
int32 = DType(3)
uint8 = DType(4)
int16 = DType(5)
int8 = DType(6)
int64 = DType(9)
bool = DType(10)
bfloat16 = DType(14)
uint16 = DType(17)
float16 = DType(19)
uint32 = DType(22)
uint64 = DType(23)

# type aliases
double = float64
half = float16

standard_promotion_table = {
    (ivy.int8, ivy.int8): ivy.int8,
    (ivy.int8, ivy.int16): ivy.int16,
    (ivy.int8, ivy.int32): ivy.int32,
    (ivy.int8, ivy.int64): ivy.int64,
    (ivy.int16, ivy.int8): ivy.int16,
    (ivy.int16, ivy.int16): ivy.int16,
    (ivy.int16, ivy.int32): ivy.int32,
    (ivy.int16, ivy.int64): ivy.int64,
    (ivy.int32, ivy.int8): ivy.int32,
    (ivy.int32, ivy.int16): ivy.int32,
    (ivy.int32, ivy.int32): ivy.int32,
    (ivy.int32, ivy.int64): ivy.int64,
    (ivy.int64, ivy.int8): ivy.int64,
    (ivy.int64, ivy.int16): ivy.int64,
    (ivy.int64, ivy.int32): ivy.int64,
    (ivy.int64, ivy.int64): ivy.int64,
    (ivy.uint8, ivy.uint8): ivy.uint8,
    (ivy.uint8, ivy.uint16): ivy.uint16,
    (ivy.uint8, ivy.uint32): ivy.uint32,
    (ivy.uint8, ivy.uint64): ivy.uint64,
    (ivy.uint16, ivy.uint8): ivy.uint16,
    (ivy.uint16, ivy.uint16): ivy.uint16,
    (ivy.uint16, ivy.uint32): ivy.uint32,
    (ivy.uint16, ivy.uint64): ivy.uint64,
    (ivy.uint32, ivy.uint8): ivy.uint32,
    (ivy.uint32, ivy.uint16): ivy.uint32,
    (ivy.uint32, ivy.uint32): ivy.uint32,
    (ivy.uint32, ivy.uint64): ivy.uint64,
    (ivy.uint64, ivy.uint8): ivy.uint64,
    (ivy.uint64, ivy.uint16): ivy.uint64,
    (ivy.uint64, ivy.uint32): ivy.uint64,
    (ivy.uint64, ivy.uint64): ivy.uint64,
    (ivy.int8, ivy.uint8): ivy.int16,
    (ivy.int8, ivy.uint16): ivy.int32,
    (ivy.int8, ivy.uint32): ivy.int64,
    (ivy.int8, ivy.uint64): ivy.float64,
    (ivy.int16, ivy.uint8): ivy.int16,
    (ivy.int16, ivy.uint16): ivy.int32,
    (ivy.int16, ivy.uint32): ivy.int64,
    (ivy.int16, ivy.uint64): ivy.float64,
    (ivy.int32, ivy.uint8): ivy.int32,
    (ivy.int32, ivy.uint16): ivy.int32,
    (ivy.int32, ivy.uint32): ivy.int64,
    (ivy.int32, ivy.uint64): ivy.float64,
    (ivy.int64, ivy.uint8): ivy.int64,
    (ivy.int64, ivy.uint16): ivy.int64,
    (ivy.int64, ivy.uint32): ivy.int64,
    (ivy.int64, ivy.uint64): ivy.float64,
    (ivy.uint8, ivy.int8): ivy.int16,
    (ivy.uint8, ivy.int16): ivy.int16,
    (ivy.uint8, ivy.int32): ivy.int32,
    (ivy.uint8, ivy.int64): ivy.int64,
    (ivy.uint16, ivy.int8): ivy.int32,
    (ivy.uint16, ivy.int16): ivy.int32,
    (ivy.uint16, ivy.int32): ivy.int32,
    (ivy.uint16, ivy.int64): ivy.int64,
    (ivy.uint32, ivy.int8): ivy.int64,
    (ivy.uint32, ivy.int16): ivy.int64,
    (ivy.uint32, ivy.int32): ivy.int64,
    (ivy.uint32, ivy.int64): ivy.int64,
    (ivy.uint64, ivy.int8): ivy.float64,
    (ivy.uint64, ivy.int16): ivy.float64,
    (ivy.uint64, ivy.int32): ivy.float64,
    (ivy.uint64, ivy.int64): ivy.float64,
    (ivy.float16, ivy.float16): ivy.float16,
    (ivy.float16, ivy.float32): ivy.float32,
    (ivy.float16, ivy.float64): ivy.float64,
    (ivy.float32, ivy.float16): ivy.float32,
    (ivy.float32, ivy.float32): ivy.float32,
    (ivy.float32, ivy.float64): ivy.float64,
    (ivy.float64, ivy.float16): ivy.float64,
    (ivy.float64, ivy.float32): ivy.float64,
    (ivy.float64, ivy.float64): ivy.float64,
    (ivy.bool, ivy.bool): ivy.bool,
}

extra_promotion_table = {
    (ivy.int8, ivy.float16): ivy.float16,
    (ivy.int8, ivy.float32): ivy.float32,
    (ivy.int8, ivy.float64): ivy.float64,
    (ivy.int16, ivy.float16): ivy.float32,
    (ivy.int16, ivy.float32): ivy.float32,
    (ivy.int16, ivy.float64): ivy.float64,
    (ivy.int32, ivy.float16): ivy.float64,
    (ivy.int32, ivy.float32): ivy.float64,
    (ivy.int32, ivy.float64): ivy.float64,
    (ivy.int64, ivy.float16): ivy.float64,
    (ivy.int64, ivy.float32): ivy.float64,
    (ivy.int64, ivy.float64): ivy.float64,
    (ivy.uint8, ivy.float16): ivy.float16,
    (ivy.uint8, ivy.float32): ivy.float32,
    (ivy.uint8, ivy.float64): ivy.float64,
    (ivy.uint16, ivy.float16): ivy.float32,
    (ivy.uint16, ivy.float32): ivy.float32,
    (ivy.uint16, ivy.float64): ivy.float64,
    (ivy.uint32, ivy.float16): ivy.float64,
    (ivy.uint32, ivy.float32): ivy.float64,
    (ivy.uint32, ivy.float64): ivy.float64,
    (ivy.uint64, ivy.float16): ivy.float64,
    (ivy.uint64, ivy.float32): ivy.float64,
    (ivy.uint64, ivy.float64): ivy.float64,
    (ivy.float16, ivy.int8): ivy.float16,
    (ivy.float16, ivy.int16): ivy.float32,
    (ivy.float16, ivy.int32): ivy.float64,
    (ivy.float16, ivy.int64): ivy.float64,
    (ivy.float32, ivy.int8): ivy.float32,
    (ivy.float32, ivy.int16): ivy.float32,
    (ivy.float32, ivy.int32): ivy.float64,
    (ivy.float32, ivy.int64): ivy.float64,
    (ivy.float64, ivy.int8): ivy.float64,
    (ivy.float64, ivy.int16): ivy.float64,
    (ivy.float64, ivy.int32): ivy.float64,
    (ivy.float64, ivy.int64): ivy.float64,
    (ivy.float16, ivy.uint8): ivy.float16,
    (ivy.float16, ivy.uint16): ivy.float32,
    (ivy.float16, ivy.uint32): ivy.float64,
    (ivy.float32, ivy.uint8): ivy.float32,
    (ivy.float32, ivy.uint16): ivy.float32,
    (ivy.float32, ivy.uint32): ivy.float64,
    (ivy.float32, ivy.uint64): ivy.float64,
    (ivy.float64, ivy.uint8): ivy.float64,
    (ivy.float64, ivy.uint16): ivy.float64,
    (ivy.float64, ivy.uint32): ivy.float64,
    (ivy.float64, ivy.uint64): ivy.float64,
    (ivy.bfloat16, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint8): ivy.bfloat16,
    (ivy.uint8, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint16): ivy.bfloat16,
    (ivy.uint16, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint32): ivy.bfloat16,
    (ivy.uint32, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.uint64): ivy.bfloat16,
    (ivy.uint64, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int8): ivy.bfloat16,
    (ivy.int8, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int16): ivy.bfloat16,
    (ivy.int16, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int32): ivy.bfloat16,
    (ivy.int32, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.int64): ivy.bfloat16,
    (ivy.int64, ivy.bfloat16): ivy.bfloat16,
    (ivy.bfloat16, ivy.float16): ivy.float32,
    (ivy.float16, ivy.bfloat16): ivy.float32,
    (ivy.bfloat16, ivy.float32): ivy.float32,
    (ivy.float32, ivy.bfloat16): ivy.float32,
    (ivy.bfloat16, ivy.float64): ivy.float64,
    (ivy.float64, ivy.bfloat16): ivy.float64,
}

# tensorflow data type promotion
tensorflow_promotion_table = {**standard_promotion_table, **extra_promotion_table}


@handle_exceptions
def promote_tensorflow_types(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
    *,
    standard_promotion: bool = False,
) -> ivy.Dtype:
    """
    Promotes the datatypes type1 and type2, returning the data type they promote to

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote
    standard_promotion
        whether to only use the standard promotion rules

    Returns
    -------
    ret
        The type that both input types promote to
    """
    try:
        if standard_promotion:
            ret = standard_promotion_table[
                (ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))
            ]
        else:
            ret = tensorflow_promotion_table[
                (ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))
            ]
    except KeyError:
        raise ivy.exceptions.IvyException("these dtypes are not type promotable")
    return ret


@handle_exceptions
def promote_types_of_tensorflow_inputs(
    x1: Union[ivy.Array, Number, Iterable[Number]],
    x2: Union[ivy.Array, Number, Iterable[Number]],
    /,
    *,
    standard_promotion: bool = False,
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Promotes the dtype of the given native array inputs to a common dtype
    based on tensorflow-specific type promotion rules. While passing float or
    integer values or any other non-array input to this function, it should be
    noted that the return will be an array-like object. Therefore, outputs
    from this function should be used as inputs only for those functions that
    expect an array-like or tensor-like objects, otherwise it might give
    unexpected results.
    """
    if (hasattr(x1, "dtype") and hasattr(x2, "dtype")) or (
        not hasattr(x1, "dtype") and not hasattr(x2, "dtype")
    ):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2)
        promoted = promote_tensorflow_types(
            x1.dtype, x2.dtype, standard_promotion=standard_promotion
        )
        x1 = ivy.asarray(x1, dtype=promoted)
        x2 = ivy.asarray(x2, dtype=promoted)
    elif hasattr(x1, "dtype"):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2, dtype=x1.dtype)
    else:
        x1 = ivy.asarray(x1, dtype=x2.dtype)
        x2 = ivy.asarray(x2)
    return x1, x2


from . import dtypes
from .dtypes import DType, as_dtype, cast
from . import ragged
from .ragged import *
from . import tensor
from .tensor import EagerTensor
from . import keras
from . import linalg
from . import math
from . import nest
from . import nn
from . import quantization
from . import random
from . import raw_ops
from . import sets
from . import signal
from . import sparse
from . import general_functions
from .general_functions import *
