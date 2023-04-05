# flake8: noqa
import ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.frontends.numpy import (
    float32,
    float64,
    float16,
    uint8,
    int32,
    int8,
    int64,
    bool_,
)

bool = bool

from numbers import Number
from typing import Union, Tuple, Iterable

from ivy.utils.backend.handler import _FrontendDictHandler

with _FrontendDictHandler() as importer:

    mxnet_promotion_table = {
        (ivy.bool, ivy.bool): ivy.bool,
        (ivy.bool, ivy.int8): ivy.int8,
        (ivy.bool, ivy.int32): ivy.int32,
        (ivy.bool, ivy.int64): ivy.int64,
        (ivy.bool, ivy.uint8): ivy.uint8,
        (ivy.bool, ivy.bfloat16): ivy.bfloat16,
        (ivy.bool, ivy.float16): ivy.float16,
        (ivy.bool, ivy.float32): ivy.float32,
        (ivy.bool, ivy.float64): ivy.float64,
        (ivy.bool, ivy.bool): ivy.bool,
        (ivy.int8, ivy.bool): ivy.int8,
        (ivy.int8, ivy.int8): ivy.int8,
        (ivy.int8, ivy.int32): ivy.int32,
        (ivy.int8, ivy.int64): ivy.int64,
        (ivy.int32, ivy.bool): ivy.int32,
        (ivy.int32, ivy.int8): ivy.int32,
        (ivy.int32, ivy.int32): ivy.int32,
        (ivy.int32, ivy.int64): ivy.int64,
        (ivy.int64, ivy.bool): ivy.int64,
        (ivy.int64, ivy.int8): ivy.int64,
        (ivy.int64, ivy.int32): ivy.int64,
        (ivy.int64, ivy.int64): ivy.int64,
        (ivy.uint8, ivy.bool): ivy.uint8,
        (ivy.uint8, ivy.uint8): ivy.uint8,
        (ivy.int32, ivy.uint8): ivy.int32,
        (ivy.int64, ivy.uint8): ivy.int64,
        (ivy.uint8, ivy.int32): ivy.int32,
        (ivy.uint8, ivy.int64): ivy.int64,
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
        (ivy.int8, ivy.float16): ivy.float16,
        (ivy.float16, ivy.int8): ivy.float16,
        (ivy.int8, ivy.float32): ivy.float32,
        (ivy.float32, ivy.int8): ivy.float32,
        (ivy.int8, ivy.float64): ivy.float64,
        (ivy.float64, ivy.int8): ivy.float64,
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
        (ivy.bfloat16, ivy.bfloat16): ivy.bfloat16,
        (ivy.bfloat16, ivy.uint8): ivy.bfloat16,
        (ivy.uint8, ivy.bfloat16): ivy.bfloat16,
        (ivy.bfloat16, ivy.int8): ivy.bfloat16,
        (ivy.int8, ivy.bfloat16): ivy.bfloat16,
        (ivy.bfloat16, ivy.float32): ivy.float32,
        (ivy.float32, ivy.bfloat16): ivy.float32,
        (ivy.bfloat16, ivy.float64): ivy.float64,
        (ivy.float64, ivy.bfloat16): ivy.float64,
    }


@handle_exceptions
def promote_types_mxnet(
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
        ret = mxnet_promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]
    except KeyError:
        raise ivy.utils.exceptions.IvyException("these dtypes are not type promotable")
    return ret


@handle_exceptions
def promote_types_of_mxnet_inputs(
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
        promoted = promote_types_mxnet(x1.dtype, x2.dtype)
        x1 = ivy.asarray(x1, dtype=promoted)
        x2 = ivy.asarray(x2, dtype=promoted)
    return x1, x2


from . import random
from . import ndarray
from . import linalg
from .linalg import *
from . import mathematical_functions
from .mathematical_functions import *
from . import creation
from .creation import *
from . import symbol
from .symbol import *
