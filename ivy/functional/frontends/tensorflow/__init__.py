# flake8: noqa
# local
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
)
from ivy.exceptions import handle_exceptions
import ivy
from numbers import Number
from typing import Union, Tuple, Iterable


standard_promotion_table = {
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
    (int8, uint64): float64,
    (int16, uint8): int16,
    (int16, uint16): int32,
    (int16, uint32): int64,
    (int16, uint64): float64,
    (int32, uint8): int32,
    (int32, uint16): int32,
    (int32, uint32): int64,
    (int32, uint64): float64,
    (int64, uint8): int64,
    (int64, uint16): int64,
    (int64, uint32): int64,
    (int64, uint64): float64,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint16, int8): int32,
    (uint16, int16): int32,
    (uint16, int32): int32,
    (uint16, int64): int64,
    (uint32, int8): int64,
    (uint32, int16): int64,
    (uint32, int32): int64,
    (uint32, int64): int64,
    (uint64, int8): float64,
    (uint64, int16): float64,
    (uint64, int32): float64,
    (uint64, int64): float64,
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

extra_promotion_table = {
    (int8, float16): float16,
    (int8, float32): float32,
    (int8, float64): float64,
    (int16, float16): float32,
    (int16, float32): float32,
    (int16, float64): float64,
    (int32, float16): float64,
    (int32, float32): float64,
    (int32, float64): float64,
    (int64, float16): float64,
    (int64, float32): float64,
    (int64, float64): float64,
    (uint8, float16): float16,
    (uint8, float32): float32,
    (uint8, float64): float64,
    (uint16, float16): float32,
    (uint16, float32): float32,
    (uint16, float64): float64,
    (uint32, float16): float64,
    (uint32, float32): float64,
    (uint32, float64): float64,
    (uint64, float16): float64,
    (uint64, float32): float64,
    (uint64, float64): float64,
    (float16, int8): float16,
    (float16, int16): float32,
    (float16, int32): float64,
    (float16, int64): float64,
    (float32, int8): float32,
    (float32, int16): float32,
    (float32, int32): float64,
    (float32, int64): float64,
    (float64, int8): float64,
    (float64, int16): float64,
    (float64, int32): float64,
    (float64, int64): float64,
    (float16, uint8): float16,
    (float16, uint16): float32,
    (float16, uint32): float64,
    (float32, uint8): float32,
    (float32, uint16): float32,
    (float32, uint32): float64,
    (float32, uint64): float64,
    (float64, uint8): float64,
    (float64, uint16): float64,
    (float64, uint32): float64,
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


from . import ragged
from .ragged import *
from . import tensor
from .tensor import *
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
from . import dtypes
from .general_functions import *
