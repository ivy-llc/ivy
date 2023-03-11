# global
from typing import Union, Optional

import paddle
# local
import ivy
from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_dtypes, with_unsupported_device_and_dtypes


def add(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    x1, x2 = x1.data, x2.data
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return paddle.add(x1, x2)


def bitwise_xor(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.bitwise_xor(x1, x2)


def expm1(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.expm1(x)


def bitwise_invert(
    x: Union[int, bool, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.bitwise_not(x)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def isfinite(
        x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.isfinite(x)


def isinf(
    x: paddle.Tensor,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if detect_negative and detect_positive:
        return paddle.isinf(x)

    if detect_negative:
        return paddle.equal(x, paddle.to_tensor([float('-inf')]))

    if detect_positive:
        return paddle.equal(x, paddle.to_tensor([float('inf')]))

    # TODO: What should happen if both detect positive and detect negative are False?
    raise IvyNotImplementedException()


def equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.equal(x1, x2)


def less_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.less_equal(x1, x2)


def bitwise_and(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.bitwise_and(x1, x2)


def ceil(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.ceil(x)


def floor(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.floor(x)


def asin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return asin(x)


def asinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.asinh(x)


def sign(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.sign(x)


def sqrt(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.sqrt(x)


def cosh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.cosh(x)


def log10(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.log10(x)


def log2(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.log2(x)


def log1p(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.log1p(x)


def isnan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.isnan(x)


def less(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.less_than(x1, x2)


def multiply(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    x1, x2 = x1.data, x2.data

    return paddle.multiply(x1, x2)


def cos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.cos(x)


def logical_not(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.logical_not(x)


def divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    x1, x2 = x1.data, x2.data

    return paddle.divide(x1, x2)


def greater(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.greater_than(x1, x2)


def greater_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.greater_equal(x1, x2)


def acos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.acos(x)


def logical_xor(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return logical_xor(x1, x2)


def logical_and(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return logical_and(x1, x2)


def logical_or(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return logical_or(x1, x2)


def acosh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.acosh(x)


def sin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.sin(x)


def negative(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.neg(x)


def not_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.not_equal(x1, x2)


def tanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.tanh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16", "float16", "complex64", "complex128")}}, backend_version
)
def floor_divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.floor(x1 / x2)


def bitwise_or(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.bitwise_or(x1, x2)


def sinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.sinh(x)


def positive(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return logical_not(negative(x))


def square(
        x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.square(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int8", "int16", "int32", "int64", "uint8", "uint16",
                                 "bfloat16", "float16", "complex64", "complex128", "bool")}}, backend_version
)
def pow(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    return paddle.pow(x1.data, x2.data)


def round(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.round(x)


def trunc(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.trunc(x)


def abs(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.abs(x)


# TODO: Is this cumsum or just sum?
def logaddexp(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def tan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.tan(x)


def atan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.atan(x)


def atan2(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.atan2(x1, x2)


def log(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.log(x)


def exp(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.exp(x)


def subtract(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    x1, x2 = x1.data, x2.data
    
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return paddle.subtract(x1, x2)


def remainder(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = paddle.where(res >= 0, paddle.floor(res), paddle.ceil(res))
        diff = paddle.to_tensor(res - res_floored, dtype=res.dtype)
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return paddle.round(diff * x2).cast(x1.dtype)
    return paddle.remainder(x1, x2)


def atanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.atanh(x)


def bitwise_right_shift(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    ret_dtype = ivy.promote_types(x1.dtype, x2.dtype)
    x1, x2 = ivy.broadcast_arrays(x1.cast('float64'), x2.cast('float64'))

    return paddle.floor(x1.data / 2**x2.data).astype(ret_dtype) 

def bitwise_left_shift(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = ivy.promote_types(x1.dtype, x2.dtype)
    x1, x2 = ivy.broadcast_arrays(x1.cast('float64'), x2.cast('float64'))

    return paddle.floor(x1.data * 2**x2.data).astype(ret_dtype) 


# Extra #
# ------#


def erf(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return paddle.erf(x)


# TODO: What does use_where do?
def minimum(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.minimum(x1, x2)


# TODO: What does use_where do?
def maximum(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if use_where:
        return paddle.where(x1 >= x2, x1, x2)
    return paddle.maximum(x1, x2)


def reciprocal(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.reciprocal(x)


def deg2rad(
        x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.deg2rad(x)


def rad2deg(
        x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.rad2deg(x)


def trunc_divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.trunc(paddle.divide(x1, x2))


def isreal(
        x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle.equal(paddle.imag(x), paddle.zeros_like(paddle.imag(x)))
    else:
        return paddle.ones_like(x, dtype='bool')
