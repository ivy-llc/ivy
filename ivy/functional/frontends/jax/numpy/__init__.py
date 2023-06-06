# global
from numbers import Number
from typing import Union, Tuple, Iterable


# local
import ivy
from ivy.utils.exceptions import handle_exceptions
import ivy.functional.frontends.jax as jax_frontend


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

# jax-numpy casting table
jax_numpy_casting_table = {
    _bool: [
        _bool,
        _int8,
        _int16,
        _int32,
        _int64,
        _uint8,
        _uint16,
        _uint32,
        _uint64,
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
        _bfloat16,
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
        _bfloat16,
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
    _int32: [
        _int32,
        _int64,
        _float64,
        _complex128,
    ],
    _int64: [
        _int64,
        _float64,
        _complex128,
    ],
    _uint8: [
        _int16,
        _int32,
        _int64,
        _uint8,
        _uint16,
        _uint32,
        _uint64,
        _float16,
        _float32,
        _float64,
        _complex64,
        _complex128,
        _bfloat16,
    ],
    _uint16: [
        _int32,
        _int64,
        _uint16,
        _uint32,
        _uint64,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
    _uint32: [
        _int64,
        _uint32,
        _uint64,
        _float64,
        _complex128,
    ],
    _uint64: [
        _uint64,
        _float64,
        _complex128,
    ],
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
    _float64: [
        _float64,
        _complex128,
    ],
    _complex64: [_complex64, ivy.complex128],
    _complex128: [_complex128],
    _bfloat16: [
        _bfloat16,
        _float32,
        _float64,
        _complex64,
        _complex128,
    ],
}


# jax-numpy type promotion table
# data type promotion
jax_promotion_table = {
    (_bool, _bool): _bool,
    (_bool, _uint8): _uint8,
    (_bool, _uint16): _uint16,
    (_bool, _uint32): _uint32,
    (_bool, _uint64): _uint64,
    (_bool, _int8): _int8,
    (_bool, _int16): _int16,
    (_bool, _int32): _int32,
    (_bool, _int64): _int64,
    (_bool, _bfloat16): _bfloat16,
    (_bool, _float16): _float16,
    (_bool, _float32): _float32,
    (_bool, _float64): _float64,
    (_bool, _complex64): _complex64,
    (_bool, _complex128): _complex128,
    (_uint8, _bool): _uint8,
    (_uint8, _uint8): _uint8,
    (_uint8, _uint16): _uint16,
    (_uint8, _uint32): _uint32,
    (_uint8, _uint64): _uint64,
    (_uint8, _int8): _int16,
    (_uint8, _int16): _int16,
    (_uint8, _int32): _int32,
    (_uint8, _int64): _int64,
    (_uint8, _bfloat16): _bfloat16,
    (_uint8, _float16): _float16,
    (_uint8, _float32): _float32,
    (_uint8, _float64): _float64,
    (_uint8, _complex64): _complex64,
    (_uint8, _complex128): _complex128,
    (_uint16, _bool): _uint16,
    (_uint16, _uint8): _uint16,
    (_uint16, _uint16): _uint16,
    (_uint16, _uint32): _uint32,
    (_uint16, _uint64): _uint64,
    (_uint16, _int8): _int32,
    (_uint16, _int16): _int32,
    (_uint16, _int32): _int32,
    (_uint16, _int64): _int64,
    (_uint16, _bfloat16): _bfloat16,
    (_uint16, _float16): _float16,
    (_uint16, _float32): _float32,
    (_uint16, _float64): _float64,
    (_uint16, _complex64): _complex64,
    (_uint16, _complex128): _complex128,
    (_uint32, _bool): _uint32,
    (_uint32, _uint8): _uint32,
    (_uint32, _uint16): _uint32,
    (_uint32, _uint32): _uint32,
    (_uint32, _uint64): _uint64,
    (_uint32, _int8): _int64,
    (_uint32, _int16): _int64,
    (_uint32, _int32): _int64,
    (_uint32, _int64): _int64,
    (_uint32, _bfloat16): _bfloat16,
    (_uint32, _float16): _float16,
    (_uint32, _float32): _float32,
    (_uint32, _float64): _float64,
    (_uint32, _complex64): _complex64,
    (_uint32, _complex128): _complex128,
    (_uint64, _bool): _uint64,
    (_uint64, _uint8): _uint64,
    (_uint64, _uint16): _uint64,
    (_uint64, _uint32): _uint64,
    (_uint64, _uint64): _uint64,
    (_uint64, _int8): _float64,
    (_uint64, _int16): _float64,
    (_uint64, _int32): _float64,
    (_uint64, _int64): _float64,
    (_uint64, _bfloat16): _bfloat16,
    (_uint64, _float16): _float16,
    (_uint64, _float32): _float32,
    (_uint64, _float64): _float64,
    (_uint64, _complex64): _complex64,
    (_uint64, _complex128): _complex128,
    (_int8, _bool): _int8,
    (_int8, _uint8): _int16,
    (_int8, _uint16): _int32,
    (_int8, _uint32): _int64,
    (_int8, _uint64): _float64,
    (_int8, _int8): _int8,
    (_int8, _int16): _int16,
    (_int8, _int32): _int32,
    (_int8, _int64): _int64,
    (_int8, _bfloat16): _bfloat16,
    (_int8, _float16): _float16,
    (_int8, _float32): _float32,
    (_int8, _float64): _float64,
    (_int8, _complex64): _complex64,
    (_int8, _complex128): _complex128,
    (_int16, _bool): _int16,
    (_int16, _uint8): _int16,
    (_int16, _uint16): _int32,
    (_int16, _uint32): _int64,
    (_int16, _uint64): _float64,
    (_int16, _int8): _int16,
    (_int16, _int16): _int16,
    (_int16, _int32): _int32,
    (_int16, _int64): _int64,
    (_int16, _bfloat16): _bfloat16,
    (_int16, _float16): _float16,
    (_int16, _float32): _float32,
    (_int16, _float64): _float64,
    (_int16, _complex64): _complex64,
    (_int16, _complex128): _complex128,
    (_int32, _bool): _int32,
    (_int32, _uint8): _int32,
    (_int32, _uint16): _int32,
    (_int32, _uint32): _int64,
    (_int32, _uint64): _float64,
    (_int32, _int8): _int32,
    (_int32, _int16): _int32,
    (_int32, _int32): _int32,
    (_int32, _int64): _int64,
    (_int32, _bfloat16): _bfloat16,
    (_int32, _float16): _float16,
    (_int32, _float32): _float32,
    (_int32, _float64): _float64,
    (_int32, _complex64): _complex64,
    (_int32, _complex128): _complex128,
    (_int64, _bool): _int64,
    (_int64, _uint8): _int64,
    (_int64, _uint16): _int64,
    (_int64, _uint32): _int64,
    (_int64, _uint64): _float64,
    (_int64, _int8): _int64,
    (_int64, _int16): _int64,
    (_int64, _int32): _int64,
    (_int64, _int64): _int64,
    (_int64, _bfloat16): _bfloat16,
    (_int64, _float16): _float16,
    (_int64, _float32): _float32,
    (_int64, _float64): _float64,
    (_int64, _complex64): _complex64,
    (_int64, _complex128): _complex128,
    (_bfloat16, _bool): _bfloat16,
    (_bfloat16, _uint8): _bfloat16,
    (_bfloat16, _uint16): _bfloat16,
    (_bfloat16, _uint32): _bfloat16,
    (_bfloat16, _uint64): _bfloat16,
    (_bfloat16, _int8): _bfloat16,
    (_bfloat16, _int16): _bfloat16,
    (_bfloat16, _int32): _bfloat16,
    (_bfloat16, _int64): _bfloat16,
    (_bfloat16, _bfloat16): _bfloat16,
    (_bfloat16, _float16): _float32,
    (_bfloat16, _float32): _float32,
    (_bfloat16, _float64): _float64,
    (_bfloat16, _complex64): _complex64,
    (_bfloat16, _complex128): _complex128,
    (_float16, _bool): _float16,
    (_float16, _uint8): _float16,
    (_float16, _uint16): _float16,
    (_float16, _uint32): _float16,
    (_float16, _uint64): _float16,
    (_float16, _int8): _float16,
    (_float16, _int16): _float16,
    (_float16, _int32): _float16,
    (_float16, _int64): _float16,
    (_float16, _bfloat16): _float32,
    (_float16, _float16): _float16,
    (_float16, _float32): _float32,
    (_float16, _float64): _float64,
    (_float16, _complex64): _complex64,
    (_float16, _complex128): _complex128,
    (_float32, _bool): _float32,
    (_float32, _uint8): _float32,
    (_float32, _uint16): _float32,
    (_float32, _uint32): _float32,
    (_float32, _uint64): _float32,
    (_float32, _int8): _float32,
    (_float32, _int16): _float32,
    (_float32, _int32): _float32,
    (_float32, _int64): _float32,
    (_float32, _bfloat16): _float32,
    (_float32, _float16): _float32,
    (_float32, _float32): _float32,
    (_float32, _float64): _float64,
    (_float32, _complex64): _complex64,
    (_float32, _complex128): _complex128,
    (_float64, _bool): _float64,
    (_float64, _uint8): _float64,
    (_float64, _uint16): _float64,
    (_float64, _uint32): _float64,
    (_float64, _uint64): _float64,
    (_float64, _int8): _float64,
    (_float64, _int16): _float64,
    (_float64, _int32): _float64,
    (_float64, _int64): _float64,
    (_float64, _bfloat16): _float64,
    (_float64, _float16): _float64,
    (_float64, _float32): _float64,
    (_float64, _float64): _float64,
    (_float64, _complex64): _complex128,
    (_float64, _complex128): _complex128,
    (_complex64, _bool): _complex64,
    (_complex64, _int8): _complex64,
    (_complex64, _int16): _complex64,
    (_complex64, _int32): _complex64,
    (_complex64, _int64): _complex64,
    (_complex64, _uint8): _complex64,
    (_complex64, _uint16): _complex64,
    (_complex64, _uint32): _complex64,
    (_complex64, _uint64): _complex64,
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
}


dtype_replacement_dict = {
    _int64: _int32,
    _uint64: _uint32,
    _float64: _float32,
    _complex128: _complex64,
}


@handle_exceptions
def promote_types_jax(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
) -> ivy.Dtype:
    """
    Promote the datatypes type1 and type2, returning the data type they promote to.

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
    Promote the dtype of the given native array inputs to a common dtype based on type
    promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
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
