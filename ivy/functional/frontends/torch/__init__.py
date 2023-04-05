# flake8: noqa

import ivy
from ivy.utils.exceptions import handle_exceptions

# global
from numbers import Number
from typing import Union, Tuple, Iterable

from ivy.utils.backend.handler import _FrontendImporter

with _FrontendImporter() as importer:

    # type aliases
    char = ivy.int8
    short = ivy.int16
    int = ivy.int32
    long = ivy.int64
    half = ivy.float16
    float = ivy.float32
    double = ivy.float64

    # data type promotion
    torch_promotion_table = {
        (ivy.uint8, ivy.uint8): ivy.uint8,
        (ivy.uint8, ivy.int8): ivy.int16,
        (ivy.uint8, ivy.int16): ivy.int16,
        (ivy.uint8, ivy.int32): ivy.int32,
        (ivy.uint8, ivy.int64): ivy.int64,
        (ivy.uint8, ivy.float16): ivy.float16,
        (ivy.uint8, ivy.float32): ivy.float32,
        (ivy.uint8, ivy.float64): ivy.float64,
        (ivy.uint8, bool): ivy.uint8,
        (ivy.uint8, ivy.bfloat16): ivy.bfloat16,
        (ivy.uint8, ivy.complex64): ivy.complex64,
        (ivy.uint8, ivy.complex128): ivy.complex128,
        (ivy.int8, ivy.uint8): ivy.int16,
        (ivy.int8, ivy.int8): ivy.int8,
        (ivy.int8, ivy.int16): ivy.int16,
        (ivy.int8, ivy.int32): ivy.int32,
        (ivy.int8, ivy.int64): ivy.int64,
        (ivy.int8, ivy.float16): ivy.float16,
        (ivy.int8, ivy.float32): ivy.float32,
        (ivy.int8, ivy.float64): ivy.float64,
        (ivy.int8, bool): ivy.int8,
        (ivy.int8, ivy.bfloat16): ivy.bfloat16,
        (ivy.int8, ivy.complex64): ivy.complex64,
        (ivy.int8, ivy.complex128): ivy.complex128,
        (ivy.int16, ivy.uint8): ivy.int16,
        (ivy.int16, ivy.int8): ivy.int16,
        (ivy.int16, ivy.int16): ivy.int16,
        (ivy.int16, ivy.int32): ivy.int32,
        (ivy.int16, ivy.int64): ivy.int64,
        (ivy.int16, ivy.float16): ivy.float16,
        (ivy.int16, ivy.float32): ivy.float32,
        (ivy.int16, ivy.float64): ivy.float64,
        (ivy.int16, bool): ivy.int16,
        (ivy.int16, ivy.bfloat16): ivy.bfloat16,
        (ivy.int16, ivy.complex64): ivy.complex64,
        (ivy.int16, ivy.complex128): ivy.complex128,
        (ivy.int32, ivy.uint8): ivy.int32,
        (ivy.int32, ivy.int8): ivy.int32,
        (ivy.int32, ivy.int16): ivy.int32,
        (ivy.int32, ivy.int32): ivy.int32,
        (ivy.int32, ivy.int64): ivy.int64,
        (ivy.int32, ivy.float16): ivy.float16,
        (ivy.int32, ivy.float32): ivy.float32,
        (ivy.int32, ivy.float64): ivy.float64,
        (ivy.int32, bool): ivy.int32,
        (ivy.int32, ivy.bfloat16): ivy.bfloat16,
        (ivy.int32, ivy.complex64): ivy.complex64,
        (ivy.int32, ivy.complex128): ivy.complex128,
        (ivy.int64, ivy.uint8): ivy.int64,
        (ivy.int64, ivy.int8): ivy.int64,
        (ivy.int64, ivy.int16): ivy.int64,
        (ivy.int64, ivy.int32): ivy.int64,
        (ivy.int64, ivy.int64): ivy.int64,
        (ivy.int64, ivy.float16): ivy.float16,
        (ivy.int64, ivy.float32): ivy.float32,
        (ivy.int64, ivy.float64): ivy.float64,
        (ivy.int64, bool): ivy.int64,
        (ivy.int64, ivy.bfloat16): ivy.bfloat16,
        (ivy.int64, ivy.complex64): ivy.complex64,
        (ivy.int64, ivy.complex128): ivy.complex128,
        (ivy.float16, ivy.uint8): ivy.float16,
        (ivy.float16, ivy.int8): ivy.float16,
        (ivy.float16, ivy.int16): ivy.float16,
        (ivy.float16, ivy.int32): ivy.float16,
        (ivy.float16, ivy.int64): ivy.float16,
        (ivy.float16, ivy.float16): ivy.float16,
        (ivy.float16, ivy.float32): ivy.float32,
        (ivy.float16, ivy.float64): ivy.float64,
        (ivy.float16, bool): ivy.float16,
        (ivy.float16, ivy.bfloat16): ivy.float32,
        (ivy.float16, ivy.complex64): ivy.complex64,
        (ivy.float16, ivy.complex128): ivy.complex128,
        (ivy.float32, ivy.uint8): ivy.float32,
        (ivy.float32, ivy.int8): ivy.float32,
        (ivy.float32, ivy.int16): ivy.float32,
        (ivy.float32, ivy.int32): ivy.float32,
        (ivy.float32, ivy.int64): ivy.float32,
        (ivy.float32, ivy.float16): ivy.float32,
        (ivy.float32, ivy.float32): ivy.float32,
        (ivy.float32, ivy.float64): ivy.float64,
        (ivy.float32, bool): ivy.float32,
        (ivy.float32, ivy.bfloat16): ivy.float32,
        (ivy.float32, ivy.complex64): ivy.complex64,
        (ivy.float32, ivy.complex128): ivy.complex128,
        (ivy.float64, ivy.uint8): ivy.float64,
        (ivy.float64, ivy.int8): ivy.float64,
        (ivy.float64, ivy.int16): ivy.float64,
        (ivy.float64, ivy.int32): ivy.float64,
        (ivy.float64, ivy.int64): ivy.float64,
        (ivy.float64, ivy.float16): ivy.float64,
        (ivy.float64, ivy.float32): ivy.float64,
        (ivy.float64, ivy.float64): ivy.float64,
        (ivy.float64, bool): ivy.float64,
        (ivy.float64, ivy.bfloat16): ivy.float64,
        (ivy.float64, ivy.complex64): ivy.complex128,
        (ivy.float64, ivy.complex128): ivy.complex128,
        (bool, ivy.uint8): ivy.uint8,
        (bool, ivy.int8): ivy.int8,
        (bool, ivy.int16): ivy.int16,
        (bool, ivy.int32): ivy.int32,
        (bool, ivy.int64): ivy.int64,
        (bool, ivy.float16): ivy.float16,
        (bool, ivy.float32): ivy.float32,
        (bool, ivy.float64): ivy.float64,
        (bool, bool): bool,
        (bool, ivy.bfloat16): ivy.bfloat16,
        (bool, ivy.complex64): ivy.complex64,
        (bool, ivy.complex128): ivy.complex128,
        (ivy.bfloat16, ivy.uint8): ivy.bfloat16,
        (ivy.bfloat16, ivy.int8): ivy.bfloat16,
        (ivy.bfloat16, ivy.int16): ivy.bfloat16,
        (ivy.bfloat16, ivy.int32): ivy.bfloat16,
        (ivy.bfloat16, ivy.int64): ivy.bfloat16,
        (ivy.bfloat16, ivy.float16): ivy.float32,
        (ivy.bfloat16, ivy.float32): ivy.float32,
        (ivy.bfloat16, ivy.float64): ivy.float64,
        (ivy.bfloat16, bool): ivy.bfloat16,
        (ivy.bfloat16, ivy.bfloat16): ivy.bfloat16,
        (ivy.bfloat16, ivy.complex64): ivy.complex64,
        (ivy.bfloat16, ivy.complex128): ivy.complex128,
        (ivy.complex64, ivy.uint8): ivy.complex64,
        (ivy.complex64, ivy.int8): ivy.complex64,
        (ivy.complex64, ivy.int16): ivy.complex64,
        (ivy.complex64, ivy.int32): ivy.complex64,
        (ivy.complex64, ivy.int64): ivy.complex64,
        (ivy.complex64, ivy.float16): ivy.complex64,
        (ivy.complex64, ivy.float32): ivy.complex64,
        (ivy.complex64, ivy.float64): ivy.complex128,
        (ivy.complex64, bool): ivy.complex64,
        (ivy.complex64, ivy.bfloat16): ivy.complex64,
        (ivy.complex64, ivy.complex64): ivy.complex64,
        (ivy.complex64, ivy.complex128): ivy.complex128,
        (ivy.complex128, ivy.uint8): ivy.complex128,
        (ivy.complex128, ivy.int8): ivy.complex128,
        (ivy.complex128, ivy.int16): ivy.complex128,
        (ivy.complex128, ivy.int32): ivy.complex128,
        (ivy.complex128, ivy.int64): ivy.complex128,
        (ivy.complex128, ivy.float16): ivy.complex128,
        (ivy.complex128, ivy.float32): ivy.complex128,
        (ivy.complex128, ivy.float64): ivy.complex128,
        (ivy.complex128, bool): ivy.complex128,
        (ivy.complex128, ivy.bfloat16): ivy.complex128,
        (ivy.complex128, ivy.complex64): ivy.complex128,
        (ivy.complex128, ivy.complex128): ivy.complex128,
    }


@handle_exceptions
def promote_types_torch(
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
        ret = torch_promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]
    except KeyError:
        raise ivy.utils.exceptions.IvyException("these dtypes are not type promotable")
    return ret


@handle_exceptions
def promote_types_of_torch_inputs(
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
    # Ignore type of 0-dim arrays to mimic torch
    if (
        hasattr(x1, "shape")
        and x1.shape == ()
        and not (hasattr(x2, "shape") and x2.shape == ())
    ):
        x1 = ivy.to_scalar(x1[()])
    if (
        hasattr(x2, "shape")
        and x2.shape == ()
        and not (hasattr(x1, "shape") and x1.shape == ())
    ):
        x2 = ivy.to_scalar(x2[()])
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
        promoted = promote_types_torch(x1.dtype, x2.dtype)
        x1 = ivy.asarray(x1, dtype=promoted)
        x2 = ivy.asarray(x2, dtype=promoted)
    return x1, x2


from . import nn
from . import tensor
from .tensor import *
from . import blas_and_lapack_ops
from .blas_and_lapack_ops import *
from . import comparison_ops
from .comparison_ops import *
from . import creation_ops
from .creation_ops import *
from . import dtype
from .dtype import *
from . import indexing_slicing_joining_mutating_ops
from .indexing_slicing_joining_mutating_ops import *
from . import locally_disabling_gradient_computation
from .locally_disabling_gradient_computation import *
from . import miscellaneous_ops
from .miscellaneous_ops import *
from . import pointwise_ops
from .pointwise_ops import *
from . import random_sampling
from .random_sampling import *
from . import reduction_ops
from .reduction_ops import *
from . import spectral_ops
from .spectral_ops import *
from . import tensor_functions
from .tensor_functions import *
from . import utilities
from .utilities import *
from . import linalg

_frontend_array = tensor
