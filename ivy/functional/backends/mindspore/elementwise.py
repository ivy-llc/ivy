# global
from typing import Union, Optional

import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as np

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _cast_for_unary_op(x):
    if not isinstance(x, ms.Tensor):
        x = ms.Tensor(x)
    return x


def add(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = ops.multiply(x2, alpha)
    return ops.add(x1, x2)


add.support_native_out = True


def bitwise_xor(
    x1: Union[int, bool, ms.Tensor],
    x2: Union[int, bool, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_xor(x1, x2)


bitwise_xor.support_native_out = True


def expm1(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.expm1(x)


expm1.support_native_out = True


def bitwise_invert(
    x: Union[int, bool, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.invert(x)


bitwise_invert.support_native_out = True


def isfinite(x: ops.Tensor, /, *, out: Optional[ops.Tensor] = None) -> ops.Tensor:
    x = _cast_for_unary_op(x)
    return ops.isfinite(x)


def isinf(
    x: ms.Tensor,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if 'float' not in str(x.dtype).lower():
        x = x.astype(ms.float32)
    if detect_negative and detect_positive:
        return ms.isinf(x)
    elif detect_negative:
        return np.isneginf(x)
    elif detect_positive:
        return np.isposinf(x)
    return np.full_like(x,False, dtype=ms.bool_)



def equal(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.eq(x1, x2)


equal.support_native_out = True


def less_equal(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ms.less_equal(x1, x2)


less_equal.support_native_out = True


def bitwise_and(
    x1: Union[int, bool, ms.Tensor],
    x2: Union[int, bool, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_and(x1, x2)


bitwise_and.support_native_out = True


def ceil(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype).lower():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.ceil(x)


ceil.support_native_out = True


def floor(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.floor(x)


floor.support_native_out = True


def asin(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.asin(x)


asin.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def asinh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.asinh(x)


asinh.support_native_out = True


def sign(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.Sign()(x)


sign.support_native_out = True


def sqrt(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.sqrt(x)


sqrt.support_native_out = True


def cosh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.cosh(x)


cosh.support_native_out = True


def log10(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log10(x)


log10.support_native_out = True


def log2(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log2(x)


def log1p(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log1p(x)


log1p.support_native_out = True


def isnan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.isnan(x)


def less(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.less(x1, x2,)


less.support_native_out = True


def multiply(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.multiply(x1, x2)


multiply.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def cos(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.cos(x)


cos.support_native_out = True


def logical_not(
    x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.logical_not(x.astype(ms.bool_))


logical_not.support_native_out = True


def bitwise_or(
    x1: Union[int, bool, ms.Tensor],
    x2: Union[int, bool, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_or(x1, x2)


bitwise_or.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def sinh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.sinh(x)


sinh.support_native_out = True


def positive(
    x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.positive(x)


def square(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.square(x)


square.support_native_out = True


def pow(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.pow(x1, x2)


pow.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def round(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.round(x)


round.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def trunc(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" not in str(x.dtype):
        return ops.trunc(x)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


def abs(
    x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.abs(x)


abs.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def logaddexp(
    x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:

    x_type = str(x1.dtype).lower()
    if 'float' not in x_type:
        x1 = x1.astype(ms.float32)
    x_type = str(x2.dtype).lower()
    if 'float' not in x_type:
        x2 = x2.astype(ms.float32)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.logaddexp(x1, x2)


logaddexp.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def tan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.tan(x)


tan.support_native_out = True


def atan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.atan(x)


atan.support_native_out = True


def atan2(
    x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x_type = str(x1.dtype).lower()
    if 'float' not in x_type:
        x1 = x1.astype(ms.float32)
    x_type = str(x2.dtype).lower()
    if 'float' not in x_type:
        x2 = x2.astype(ms.float32)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.atan2(x1, x2)


atan2.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def log(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log(x)


log.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def exp(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.exp(x)


exp.support_native_out = True


def subtract(
    x1: Union[float, ms.Tensor],
    x2: Union[float, ms.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = ops.multiply(x2, alpha)
    return ops.subtract(x1, x2)


subtract.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def atanh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)

    x = _cast_for_unary_op(x)
    return ops.atanh(x)


atanh.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def erf(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    x = _cast_for_unary_op(x)
    return ops.erf(x)


erf.support_native_out = True



@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def reciprocal(
    x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.Reciprocal()(x)


reciprocal.support_native_out = True


def deg2rad(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    return ops.deg2rad(x)


deg2rad.support_native_out = True


def rad2deg(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    return ops.rad2deg(x)


rad2deg.support_native_out = True


def isreal(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.isreal(x)

