# flake8: noqa
# global
from numbers import Number
from typing import Union, Tuple, Iterable


# local
import ivy
from ivy.utils.backend.handler import _FrontendDictHandler
from ivy.utils.exceptions import handle_exceptions
import ivy.functional.frontends.jax as jax_frontend


with _FrontendDictHandler() as importer:

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
        ],
        ivy.int32: [
            ivy.int32,
            ivy.int64,
            ivy.float64,
            ivy.complex128,
        ],
        ivy.int64: [
            ivy.int64,
            ivy.float64,
            ivy.complex128,
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
        ],
        ivy.uint32: [
            ivy.int64,
            ivy.uint32,
            ivy.uint64,
            ivy.float64,
            ivy.complex128,
        ],
        ivy.uint64: [
            ivy.uint64,
            ivy.float64,
            ivy.complex128,
        ],
        ivy.float16: [
            ivy.float16,
            ivy.float32,
            ivy.float64,
            ivy.complex64,
            ivy.complex128,
        ],
        ivy.float32: [
            ivy.float32,
            ivy.float64,
            ivy.complex64,
            ivy.complex128,
        ],
        ivy.float64: [
            ivy.float64,
            ivy.complex128,
        ],
        ivy.complex64: [ivy.complex64, ivy.complex128],
        ivy.complex128: [ivy.complex128],
        ivy.bfloat16: [
            ivy.bfloat16,
            ivy.float32,
            ivy.float64,
            ivy.complex64,
            ivy.complex128,
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
        (ivy.bool, ivy.complex64): ivy.complex64,
        (ivy.bool, ivy.complex128): ivy.complex128,
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
        (ivy.uint8, ivy.complex64): ivy.complex64,
        (ivy.uint8, ivy.complex128): ivy.complex128,
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
        (ivy.uint16, ivy.complex64): ivy.complex64,
        (ivy.uint16, ivy.complex128): ivy.complex128,
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
        (ivy.uint32, ivy.complex64): ivy.complex64,
        (ivy.uint32, ivy.complex128): ivy.complex128,
        (ivy.uint64, ivy.bool): ivy.uint64,
        (ivy.uint64, ivy.uint8): ivy.uint64,
        (ivy.uint64, ivy.uint16): ivy.uint64,
        (ivy.uint64, ivy.uint32): ivy.uint64,
        (ivy.uint64, ivy.uint64): ivy.uint64,
        (ivy.uint64, ivy.int8): ivy.float64,
        (ivy.uint64, ivy.int16): ivy.float64,
        (ivy.uint64, ivy.int32): ivy.float64,
        (ivy.uint64, ivy.int64): ivy.float64,
        (ivy.uint64, ivy.bfloat16): ivy.bfloat16,
        (ivy.uint64, ivy.float16): ivy.float16,
        (ivy.uint64, ivy.float32): ivy.float32,
        (ivy.uint64, ivy.float64): ivy.float64,
        (ivy.uint64, ivy.complex64): ivy.complex64,
        (ivy.uint64, ivy.complex128): ivy.complex128,
        (ivy.int8, ivy.bool): ivy.int8,
        (ivy.int8, ivy.uint8): ivy.int16,
        (ivy.int8, ivy.uint16): ivy.int32,
        (ivy.int8, ivy.uint32): ivy.int64,
        (ivy.int8, ivy.uint64): ivy.float64,
        (ivy.int8, ivy.int8): ivy.int8,
        (ivy.int8, ivy.int16): ivy.int16,
        (ivy.int8, ivy.int32): ivy.int32,
        (ivy.int8, ivy.int64): ivy.int64,
        (ivy.int8, ivy.bfloat16): ivy.bfloat16,
        (ivy.int8, ivy.float16): ivy.float16,
        (ivy.int8, ivy.float32): ivy.float32,
        (ivy.int8, ivy.float64): ivy.float64,
        (ivy.int8, ivy.complex64): ivy.complex64,
        (ivy.int8, ivy.complex128): ivy.complex128,
        (ivy.int16, ivy.bool): ivy.int16,
        (ivy.int16, ivy.uint8): ivy.int16,
        (ivy.int16, ivy.uint16): ivy.int32,
        (ivy.int16, ivy.uint32): ivy.int64,
        (ivy.int16, ivy.uint64): ivy.float64,
        (ivy.int16, ivy.int8): ivy.int16,
        (ivy.int16, ivy.int16): ivy.int16,
        (ivy.int16, ivy.int32): ivy.int32,
        (ivy.int16, ivy.int64): ivy.int64,
        (ivy.int16, ivy.bfloat16): ivy.bfloat16,
        (ivy.int16, ivy.float16): ivy.float16,
        (ivy.int16, ivy.float32): ivy.float32,
        (ivy.int16, ivy.float64): ivy.float64,
        (ivy.int16, ivy.complex64): ivy.complex64,
        (ivy.int16, ivy.complex128): ivy.complex128,
        (ivy.int32, ivy.bool): ivy.int32,
        (ivy.int32, ivy.uint8): ivy.int32,
        (ivy.int32, ivy.uint16): ivy.int32,
        (ivy.int32, ivy.uint32): ivy.int64,
        (ivy.int32, ivy.uint64): ivy.float64,
        (ivy.int32, ivy.int8): ivy.int32,
        (ivy.int32, ivy.int16): ivy.int32,
        (ivy.int32, ivy.int32): ivy.int32,
        (ivy.int32, ivy.int64): ivy.int64,
        (ivy.int32, ivy.bfloat16): ivy.bfloat16,
        (ivy.int32, ivy.float16): ivy.float16,
        (ivy.int32, ivy.float32): ivy.float32,
        (ivy.int32, ivy.float64): ivy.float64,
        (ivy.int32, ivy.complex64): ivy.complex64,
        (ivy.int32, ivy.complex128): ivy.complex128,
        (ivy.int64, ivy.bool): ivy.int64,
        (ivy.int64, ivy.uint8): ivy.int64,
        (ivy.int64, ivy.uint16): ivy.int64,
        (ivy.int64, ivy.uint32): ivy.int64,
        (ivy.int64, ivy.uint64): ivy.float64,
        (ivy.int64, ivy.int8): ivy.int64,
        (ivy.int64, ivy.int16): ivy.int64,
        (ivy.int64, ivy.int32): ivy.int64,
        (ivy.int64, ivy.int64): ivy.int64,
        (ivy.int64, ivy.bfloat16): ivy.bfloat16,
        (ivy.int64, ivy.float16): ivy.float16,
        (ivy.int64, ivy.float32): ivy.float32,
        (ivy.int64, ivy.float64): ivy.float64,
        (ivy.int64, ivy.complex64): ivy.complex64,
        (ivy.int64, ivy.complex128): ivy.complex128,
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
        (ivy.bfloat16, ivy.complex64): ivy.complex64,
        (ivy.bfloat16, ivy.complex128): ivy.complex128,
        (ivy.float16, ivy.bool): ivy.float16,
        (ivy.float16, ivy.uint8): ivy.float16,
        (ivy.float16, ivy.uint16): ivy.float16,
        (ivy.float16, ivy.uint32): ivy.float16,
        (ivy.float16, ivy.uint64): ivy.float16,
        (ivy.float16, ivy.int8): ivy.float16,
        (ivy.float16, ivy.int16): ivy.float16,
        (ivy.float16, ivy.int32): ivy.float16,
        (ivy.float16, ivy.int64): ivy.float16,
        (ivy.float16, ivy.bfloat16): ivy.float32,
        (ivy.float16, ivy.float16): ivy.float16,
        (ivy.float16, ivy.float32): ivy.float32,
        (ivy.float16, ivy.float64): ivy.float64,
        (ivy.float16, ivy.complex64): ivy.complex64,
        (ivy.float16, ivy.complex128): ivy.complex128,
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
        (ivy.float32, ivy.complex64): ivy.complex64,
        (ivy.float32, ivy.complex128): ivy.complex128,
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
        (ivy.float64, ivy.complex64): ivy.complex128,
        (ivy.float64, ivy.complex128): ivy.complex128,
        (ivy.complex64, ivy.bool): ivy.complex64,
        (ivy.complex64, ivy.int8): ivy.complex64,
        (ivy.complex64, ivy.int16): ivy.complex64,
        (ivy.complex64, ivy.int32): ivy.complex64,
        (ivy.complex64, ivy.int64): ivy.complex64,
        (ivy.complex64, ivy.uint8): ivy.complex64,
        (ivy.complex64, ivy.uint16): ivy.complex64,
        (ivy.complex64, ivy.uint32): ivy.complex64,
        (ivy.complex64, ivy.uint64): ivy.complex64,
        (ivy.complex64, ivy.float16): ivy.complex64,
        (ivy.complex64, ivy.float32): ivy.complex64,
        (ivy.complex64, ivy.float64): ivy.complex128,
        (ivy.complex64, ivy.bfloat16): ivy.complex64,
        (ivy.complex64, ivy.complex64): ivy.complex64,
        (ivy.complex64, ivy.complex128): ivy.complex128,
        (ivy.complex128, ivy.bool): ivy.complex128,
        (ivy.complex128, ivy.int8): ivy.complex128,
        (ivy.complex128, ivy.int16): ivy.complex128,
        (ivy.complex128, ivy.int32): ivy.complex128,
        (ivy.complex128, ivy.int64): ivy.complex128,
        (ivy.complex128, ivy.uint8): ivy.complex128,
        (ivy.complex128, ivy.uint16): ivy.complex128,
        (ivy.complex128, ivy.uint32): ivy.complex128,
        (ivy.complex128, ivy.uint64): ivy.complex128,
        (ivy.complex128, ivy.float16): ivy.complex128,
        (ivy.complex128, ivy.float32): ivy.complex128,
        (ivy.complex128, ivy.float64): ivy.complex128,
        (ivy.complex128, ivy.bfloat16): ivy.complex128,
        (ivy.complex128, ivy.complex64): ivy.complex128,
        (ivy.complex128, ivy.complex128): ivy.complex128,
    }

    dtype_replacement_dict = {
        ivy.int64: ivy.int32,
        ivy.uint64: ivy.uint32,
        ivy.float64: ivy.float32,
        ivy.complex128: ivy.complex64,
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
        raise ivy.utils.exceptions.IvyException("these dtypes are not type promotable")
    return ret


def _handle_x64_promotion(d):
    if not jax_frontend.config.jax_enable_x64:
        d = dtype_replacement_dict[d] if d in dtype_replacement_dict else d
    return d


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
    type1 = ivy.default_dtype(item=x1).strip("u123456789")
    type2 = ivy.default_dtype(item=x2).strip("u123456789")
    if hasattr(x1, "dtype") and not hasattr(x2, "dtype") and type1 == type2:
        x2 = ivy.asarray(
            x2, dtype=x1.dtype, device=ivy.default_device(item=x1, as_native=False)
        )
    elif not hasattr(x1, "dtype") and hasattr(x2, "dtype") and type1 == type2:
        x1 = ivy.asarray(
            x1, dtype=x2.dtype, device=ivy.default_device(item=x2, as_native=False)
        )
    else:
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2)
        x1_type, x2_type = x1.dtype, x2.dtype
        if x1_type != x2_type:
            x1_type = _handle_x64_promotion(x1_type)
            x2_type = _handle_x64_promotion(x2_type)
            promoted = _handle_x64_promotion(promote_types_jax(x1_type, x2_type))
            x1 = ivy.asarray(x1, dtype=promoted)
            x2 = ivy.asarray(x2, dtype=promoted)
    return x1, x2


from . import fft
from . import linalg
from . import creation
from .creation import *
from .dtype import can_cast, promote_types
from .scalars import *
from . import indexing
from .indexing import *
from . import logic
from .logic import *
from . import manipulations
from .manipulations import *
from . import mathematical_functions
from .mathematical_functions import *
from . import statistical
from .statistical import *
from . import searching_sorting
from .searching_sorting import *

_frontend_array = array
