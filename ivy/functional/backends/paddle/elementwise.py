# global
from typing import Union, Optional

import paddle
# local
import ivy
from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException
from ivy.func_wrapper import with_unsupported_dtypes, with_unsupported_device_and_dtypes


def _elementwise_helper(x1, x2):
    x1, x2 = ivy.to_native(ivy.array(x1)), ivy.to_native(ivy.array(x2))
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = ivy.broadcast_arrays(x1, x2)
    return ivy.to_native(x1), ivy.to_native(x2), x1.dtype


def _complex_modulus(x):
    return paddle.sqrt(x.real()**2 + x.imag()**2)


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
        x1, x2 = x1.cast('float32'), x2.cast('float32')
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return paddle.add(x1, x2).cast(ret_dtype)


def bitwise_xor(
    x1: Union[int, bool, paddle.Tensor],
    x2: Union[int, bool, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.bitwise_xor(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
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
        return equal(x, paddle.to_tensor([float('-inf')]))

    if detect_positive:
        return equal(x, paddle.to_tensor([float('inf')]))

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
    if x1.dtype in [paddle.int8, paddle.uint8, paddle.complex64, paddle.complex128]:
        if paddle.is_complex(x1):
            return logical_and(paddle.equal(x1.real(), x2.real()), paddle.equal(x1.imag(), x2.imag()))
        return paddle.equal(x1.cast('float32'), x2.cast('float32'))
    return paddle.equal(x1, x2)


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
        return paddle.less_equal(x1.cast('float32'), x2.cast('float32'))

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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def asin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.asin(x.cast('float32')).cast(ret_dtype)
    return paddle.asin(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def asinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.asinh(x.cast('float32')).cast(ret_dtype)
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
        return paddle.less_than(x1.cast('float32'), x2.cast('float32'))

    return paddle.less_than(x1, x2)


def multiply(
    x1: Union[float, paddle.Tensor],
    x2: Union[float, paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        x1, x2 = x1.cast('float32'), x2.cast('float32')
    return paddle.multiply(x1, x2).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def cos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.cos(x.cast('float32')).cast(ret_dtype)
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
        return paddle.greater_than(x1.cast('float32'), x2.cast('float32'))

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
        return paddle.greater_equal(x1.cast('float32'), x2.cast('float32'))

    return paddle.greater_equal(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def acos(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.acos(x.cast('float32')).cast(ret_dtype)
    return paddle.acos(x)


def logical_xor(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.logical_xor(x1, x2)


def logical_and(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.logical_and(x1, x2)


def logical_or(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return paddle.logical_or(x1, x2)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def acosh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.acosh(x.cast('float32')).cast(ret_dtype)
    return paddle.acosh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def sin(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.sin(x.cast('float32')).cast(ret_dtype)
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
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def tanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.tanh(x.cast('float32')).cast(ret_dtype)
    return paddle.tanh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", "float16", "complex64", "complex128")}}, backend_version
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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def sinh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.sinh(x.cast('float32')).cast(ret_dtype)
    return paddle.sinh(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def positive(
    x: Union[float, paddle.Tensor], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x, _ = ivy.promote_types_of_inputs(x, x)
    return x.clone()


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
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    return log(add(exp(x1), exp(x2))).cast(ret_dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def tan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.tan(x.cast('float32')).cast(ret_dtype)
    return paddle.tan(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def atan(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.atan(x.cast('float32')).cast(ret_dtype)
    return paddle.atan(x)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def atan2(
    x1: paddle.Tensor, x2: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x1, x2, ret_dtype = _elementwise_helper(x1, x2)
    if x1.dtype in [paddle.int8, paddle.int16, paddle.uint8]:
        x1, x2 = x1.cast('float32'), x2.cast('float32')
    return paddle.atan2(x1, x2).cast(ret_dtype)


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


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {
        "cpu": ("uint16", "bfloat16", 'complex64', 'complex128', 'bool')}}, backend_version
)
def atanh(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64, paddle.uint8, paddle.float16]:
        ret_dtype = x.dtype
        return paddle.atanh(x.cast('float32')).cast(ret_dtype)
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
        return paddle.logical_not(x.imag().astype(bool))
    else:
        return paddle.ones_like(x, dtype='bool')
