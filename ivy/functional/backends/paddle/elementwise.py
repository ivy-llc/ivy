# global
from typing import Union, Optional

import paddle
import math

# local
import ivy
from . import backend_version
from ivy.func_wrapper import with_unsupported_dtypes, with_unsupported_device_and_dtypes


def _elementwise_helper(x1, x2):
    with ivy.ArrayMode(False):
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
        x1, x2 = ivy.broadcast_arrays(x1, x2)
    return x1, x2, x1.dtype


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def add(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.float16, paddle.bool]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    if alpha not in (1, None):
        x2 = ivy.to_native(multiply(x2, alpha))
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.add(x1, x2).cast(ret_dtype)


def bitwise_xor(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return paddle.bitwise_xor(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def expm1(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return subtract(exp(x), 1.0).cast(x.dtype)


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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
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
        return equal(x, paddle.to_tensor([float("-inf")]))

    if detect_positive:
        return equal(x, paddle.to_tensor([float("inf")]))

    return paddle.zeros(shape=x.shape, dtype=bool)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    with ivy.ArrayMode(False):
        diff = ivy.subtract(x1, x2)
        ret = ivy.logical_and(ivy.less_equal(diff, 0), ivy.greater_equal(diff, 0))
        # ret result is sufficient for all cases except where the value is +/-INF of NaN
        return ivy.where(
            ivy.isnan(diff), ~ivy.logical_or(ivy.isnan(x1), ivy.isnan(x2)), ret
        )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def less_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            if paddle.is_complex(x1):
                real = paddle.less_equal(x1.real(), x2.real())
                imag = paddle.less_equal(x1.imag(), x2.imag())
                return logical_and(real, imag)
        return paddle.less_equal(x1.cast("float32"), x2.cast("float32"))

    return paddle.less_equal(x1, x2)


def bitwise_and(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return paddle.bitwise_and(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def ceil(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.ceil(x.real()) + paddle.ceil(x.imag()) * 1j
        return paddle.ceil(x.cast("float32")).cast(x.dtype)
    return paddle.ceil(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def floor(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.floor(x.real()) + paddle.floor(x.imag()) * 1j
        return paddle.floor(x.cast("float32")).cast(x.dtype)
    return paddle.floor(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def asin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.asin(x.cast("float32")).cast(ret_dtype)
    return paddle.asin(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def asinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.asinh(x.cast("float32")).cast(ret_dtype)
    return paddle.asinh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def sign(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        return paddle.sgn(x.cast("float32")).cast(x.dtype)
    return paddle.sgn(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def sqrt(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            angle = paddle.angle(x)
            result = (paddle.cos(angle / 2) + 1j * paddle.sin(angle / 2)) * paddle.sqrt(
                paddle.abs(x)
            )
            return result
        return paddle.sqrt(x.cast("float32")).cast(x.dtype)
    return paddle.sqrt(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def cosh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.cosh(x.cast("float32")).cast(ret_dtype)
    return paddle.cosh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def log10(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    base = paddle.to_tensor(10.0)
    return divide(log(x), log(base)).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def log2(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    base = paddle.to_tensor(2.0)
    return divide(log(x), log(base)).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def log1p(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.log1p(paddle.abs(x)) + 1j * paddle.angle(x + 1)
        return paddle.log1p(x.cast("float32")).cast(x.dtype)
    return paddle.log1p(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def isnan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.logical_or(paddle.isnan(x.real()), paddle.isnan(x.imag()))
        return paddle.isnan(x.cast("float32"))
    return paddle.isnan(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def less(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            real = paddle.less_than(x1.real(), x2.real())
            imag = paddle.less_than(x1.imag(), x2.imag())
            return logical_and(real, imag)
        return paddle.less_than(x1.cast("float32"), x2.cast("float32"))

    return paddle.less_than(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def multiply(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.multiply(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def cos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.cos(x.cast("float32")).cast(ret_dtype)
    return paddle.cos(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logical_not(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.uint8, paddle.float16, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x):
            return paddle.logical_and(
                paddle.logical_not(x.real()), paddle.logical_not(x.imag())
            )
        return paddle.logical_not(x.cast("float32"))
    return paddle.logical_not(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.float16]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    if not (ivy.is_float_dtype(ret_dtype) or ivy.is_complex_dtype(ret_dtype)):
        ret_dtype = ivy.default_float_dtype(as_native=True)
    return (x1 / x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def greater(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            if paddle.is_complex(x1):
                real = paddle.greater_than(x1.real(), x2.real())
                imag = paddle.greater_than(x1.imag(), x2.imag())
                return logical_and(real, imag)
        return paddle.greater_than(x1.cast("float32"), x2.cast("float32"))

    return paddle.greater_than(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def greater_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            if paddle.is_complex(x1):
                real = paddle.greater_equal(x1.real(), x2.real())
                imag = paddle.greater_equal(x1.imag(), x2.imag())
                return logical_and(real, imag)
        return paddle.greater_equal(x1.cast("float32"), x2.cast("float32"))

    return paddle.greater_equal(x1, x2)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def acos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.acos(x.cast("float32")).cast(ret_dtype)
    return paddle.acos(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logical_xor(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if ret_dtype in [paddle.uint8, paddle.float16, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            return paddle.logical_xor(
                paddle.logical_xor(x1.real(), x2.real()),
                paddle.logical_xor(x1.imag(), x2.imag()),
            )
        return paddle.logical_xor(x1.cast("float32"), x2.cast("float32"))
    return paddle.logical_xor(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logical_and(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if ret_dtype in [paddle.uint8, paddle.float16, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            return paddle.logical_and(
                paddle.logical_and(x1.real(), x2.real()),
                paddle.logical_and(x1.imag(), x2.imag()),
            )
        return paddle.logical_and(x1.cast("float32"), x2.cast("float32"))
    return paddle.logical_and(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logical_or(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if ret_dtype in [paddle.uint8, paddle.float16, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            return paddle.logical_or(
                paddle.logical_or(x1.real(), x2.real()),
                paddle.logical_or(x1.imag(), x2.imag()),
            )
        return paddle.logical_or(x1.cast("float32"), x2.cast("float32"))
    return paddle.logical_or(x1, x2)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def acosh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.acosh(x.cast("float32")).cast(ret_dtype)
    return paddle.acosh(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def sin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.sin(x.cast("float32")).cast(ret_dtype)
    return paddle.sin(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def negative(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x, _ = ivy.promote_types_of_inputs(x, x)
    if x.dtype == paddle.bool:
        return logical_not(x)
    return paddle.neg(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def not_equal(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return logical_not(equal(x1, x2))


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def tanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.tanh(x.cast("float32")).cast(ret_dtype)
    return paddle.tanh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}},
    backend_version,
)
def floor_divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    with ivy.ArrayMode(False):
        return floor(divide(x1, x2)).cast(ret_dtype)


def bitwise_or(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return paddle.bitwise_or(x1, x2)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def sinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.sinh(x.cast("float32")).cast(ret_dtype)
    return paddle.sinh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def positive(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x, _ = ivy.promote_types_of_inputs(x, x)
    return x.clone()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def square(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return pow(x, 2).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def pow(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x1):
            # https://math.stackexchange.com/questions/476968/complex-power-of-a-complex-number
            r = paddle.abs(x1)
            theta = paddle.angle(x1)
            power = x2 * (paddle.log(r) + 1j * theta)
            result = paddle.exp(power.real()) * (
                paddle.cos(power.imag()) + 1j * paddle.sin(power.imag())
            )
            return result
        return paddle.pow(x1.cast("float64"), x2.cast("float64")).cast(ret_dtype)
    return paddle.pow(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def round(
    x: paddle.Tensor, /, *, decimals: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    def _np_round(x, decimals):
        # this is a logic to mimic np.round behaviour
        # which rounds odd numbers up and even numbers down at limits like 0.5
        eps = 1e-6 * paddle.sign(x)

        with ivy.ArrayMode(False):
            # check if the integer is even or odd
            candidate_ints = ivy.remainder(ivy.trunc(x), 2.0).astype(bool)
            # check if the fraction is exactly half
            candidate_fractions = ivy.equal(ivy.abs(ivy.subtract(x, ivy.trunc(x))), 0.5)
            x = ivy.where(
                ivy.logical_and(~candidate_ints, candidate_fractions),
                x - eps,
                x,
            )
            factor = ivy.pow(10, decimals).astype(x.dtype)
            factor_denom = ivy.where(ivy.isinf(x), 1, factor)
            return ivy.divide(paddle.round(ivy.multiply(x, factor)), factor_denom)

    x, _ = ivy.promote_types_of_inputs(x, x)
    if x.dtype not in [paddle.float32, paddle.float64]:
        if paddle.is_complex(x):
            return _np_round(x.real(), decimals) + _np_round(x.imag(), decimals) * 1j
        return _np_round(x.cast("float32"), decimals).cast(x.dtype)
    return _np_round(x, decimals)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def trunc(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.trunc(x.real()) + 1j * paddle.trunc(x.imag())
        return paddle.trunc(x.cast("float32")).cast(x.dtype)
    return paddle.trunc(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def abs(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.bool,
    ]:
        return paddle.abs(x.cast("float32")).cast(x.dtype)
    return paddle.abs(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logaddexp(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return log(add(exp(x1), exp(x2))).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def tan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.tan(x.cast("float32")).cast(ret_dtype)
    return paddle.tan(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def atan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.atan(x.cast("float32")).cast(ret_dtype)
    return paddle.atan(x)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def atan2(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.int16, paddle.uint8]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.atan2(x1, x2).cast(ret_dtype)


def log(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x):
            return paddle.log(paddle.abs(x)) + 1j * paddle.angle(x)
        return paddle.log(x.cast("float32")).cast(x.dtype)
    return paddle.log(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def exp(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return pow(math.e, x).cast(x.dtype)


def subtract(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.float16, paddle.bool]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    if alpha not in (1, None):
        x2 = ivy.to_native(multiply(x2, alpha))
        x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.subtract(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def remainder(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if not modulus:
        with ivy.ArrayMode(False):
            res = divide(x1, x2)
            res_floored = ivy.where(greater_equal(res, 0), floor(res), ceil(res))
            diff = subtract(res, res_floored).astype(res.dtype)
            return round(multiply(diff, x2)).cast(x1.dtype)

    if x1.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        x1, x2 = x1.cast("float32"), x2.cast("float32")
    return paddle.remainder(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def atanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
        paddle.uint8,
        paddle.float16,
    ]:
        ret_dtype = x.dtype
        return paddle.atanh(x.cast("float32")).cast(ret_dtype)
    return paddle.atanh(x)


def bitwise_right_shift(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return paddle.floor(x1.cast("float64") / 2 ** x2.cast("float64")).astype(ret_dtype)


def bitwise_left_shift(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return paddle.floor(x1.cast("float64") * 2 ** x2.cast("float64")).astype(ret_dtype)


# Extra #
# ------#


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "complex64", "complex128", "bool")
        }
    },
    backend_version,
)
def erf(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    # TODO: add support for complex x, supported in scipy only atm
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8]:
        return paddle.erf(x.cast("float32")).cast(x.dtype)
    return paddle.erf(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def minimum(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x1):
            use_where = True
        else:
            x1, x2 = x1.cast("float32"), x2.cast("float32")

    if use_where:
        return ivy.where(less_equal(x1, x2), x1, x2).cast(ret_dtype)

    return paddle.minimum(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def maximum(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(x1):
            use_where = True
        else:
            x1, x2 = x1.cast("float32"), x2.cast("float32")
    if use_where:
        return ivy.where(greater_equal(x1, x2), x1, x2).cast(ret_dtype)
    return paddle.maximum(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def reciprocal(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return divide(1, x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def deg2rad(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.int32, paddle.int64, paddle.bool]:
        return paddle.deg2rad(x.cast("float32")).cast(x.dtype)
    return paddle.deg2rad(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def rad2deg(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype in [paddle.int32, paddle.int64, paddle.bool]:
        return paddle.rad2deg(x.cast("float32")).cast(x.dtype)
    return paddle.rad2deg(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def trunc_divide(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return trunc(divide(x1, x2))


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def isreal(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if paddle.is_complex(x):
        return paddle.logical_not(x.imag().astype(bool))
    else:
        return paddle.ones_like(x, dtype="bool")


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def fmod(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    with ivy.ArrayMode(False):
        res = ivy.remainder(ivy.abs(x1), ivy.abs(x2))
        return ivy.where(ivy.less(x1, 0), -res, res)
