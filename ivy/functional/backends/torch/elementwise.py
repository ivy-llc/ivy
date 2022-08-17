# global
import torch
from typing import Union, Optional

# local
import ivy


def _cast_for_unary_op(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x


def _cast_for_binary_op(x1, x2, clamp=False):
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)

    if clamp:
        x2 = torch.clamp(x2, max=torch.iinfo(x1.dtype).bits - 1)
    return x1, x2


def add(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.add(x1, x2, out=out)


add.support_native_out = True


def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


def expm1(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.expm1(x, out=out)


expm1.unsupported_dtypes = ("float16",)
expm1.support_native_out = True


def bitwise_invert(
    x: Union[int, bool, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


bitwise_invert.support_native_out = True


def isfinite(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.isfinite(x)


def isinf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.isinf(x)


def equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.eq(x1, x2, out=out)


equal.support_native_out = True


def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


def ceil(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.support_native_out = True
ceil.unsupported_dtypes = ("float16",)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.support_native_out = True
floor.unsupported_dtypes = ("float16",)


def asin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.asin(x, out=out)


asin.support_native_out = True
asin.unsupported_dtypes = ("float16",)


def asinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.asinh(x, out=out)


asinh.support_native_out = True
asinh.unsupported_dtypes = ("float16",)


def sign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sign(x, out=out)


sign.support_native_out = True


def sqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sqrt(x, out=out)


sqrt.support_native_out = True
sqrt.unsupported_dtypes = ("float16",)


def cosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.cosh(x, out=out)


cosh.support_native_out = True
cosh.unsupported_dtypes = ("float16",)


def log10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log10(x, out=out)


log10.support_native_out = True
log10.unsupported_dtypes = ("float16",)


def log2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log2(x, out=out)


log2.unsupported_dtypes = ("float16",)


def log1p(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log1p(x, out=out)


log1p.support_native_out = True
log1p.unsupported_dtypes = ("float16",)


def isnan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.isnan(x)
    return ret


def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.lt(x1, x2, out=out)


less.support_native_out = True


def multiply(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.multiply(x1, x2, out=out)


multiply.support_native_out = True


def cos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.cos(x, out=out)


cos.support_native_out = True
cos.unsupported_dtypes = ("float16",)


def logical_not(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool), out=out)


logical_not.support_native_out = True


def divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = torch.div(x1, x2)
    if ivy.is_float_dtype(x1.dtype):
        ret = torch.tensor(ret, dtype=x1.dtype)
    else:
        ret = torch.tensor(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


divide.support_native_out = True


def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.greater(x1, x2, out=out)


greater.support_native_out = True


def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


def acos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.acos(x, out=out)


acos.support_native_out = True
acos.unsupported_dtypes = ("float16",)


def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_xor.support_native_out = True


def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_and.support_native_out = True


def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_or.support_native_out = True


def acosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.acosh(x, out=out)


acosh.support_native_out = True
acosh.unsupported_dtypes = ("float16",)


def sin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sin(x, out=out)


sin.support_native_out = True
sin.unsupported_dtypes = ("float16",)


def negative(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.neg(x, out=out)


negative.support_native_out = True


def not_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


def tanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tanh(x, out=out)


tanh.support_native_out = True
tanh.unsupported_dtypes = ("float16",)


def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.div(x1, x2, rounding_mode="floor", out=out)


floor_divide.support_native_out = True


def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


def sinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sinh(x, out=out)


sinh.support_native_out = True
sinh.unsupported_dtypes = ("float16",)


def positive(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    ret = torch.positive(x)
    return ret


def square(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.square(x, out=out)


square.support_native_out = True


def pow(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.pow(x1, x2, out=out)


pow.support_native_out = True


def round(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.round(x, out=out)


round.support_native_out = True
round.unsupported_dtypes = ("float16",)


def trunc(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" not in str(x.dtype):
        return torch.trunc(x, out=out)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True
trunc.unsupported_dtypes = ("float16",)


def abs(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.abs(x, out=out)


abs.support_native_out = True


def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True
logaddexp.unsupported_dtypes = ("float16",)


def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tan(x, out=out)


tan.support_native_out = True
tan.unsupported_dtypes = ("float16",)


def atan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.atan(x, out=out)


atan.support_native_out = True
atan.unsupported_dtypes = ("float16",)


def atan2(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.support_native_out = True
atan2.unsupported_dtypes = ("float16",)


def log(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log(x, out=out)


log.support_native_out = True
log.unsupported_dtypes = ("float16",)


def exp(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.exp(x, out=out)


exp.support_native_out = True
exp.unsupported_dtypes = ("float16",)


def subtract(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.subtract(x1, x2, out=out)


subtract.support_native_out = True


def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = torch.remainder(x1, x2, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


remainder.support_native_out = True


def atanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.atanh(x, out=out)


atanh.support_native_out = True
atanh.unsupported_dtypes = ("float16",)


def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2, clamp=True)
    return torch.bitwise_right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2, clamp=True)
    return torch.bitwise_left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


# Extra #
# ------#


def erf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.erf(x, out=out)


erf.support_native_out = True
erf.unsupported_dtypes = ("float16",)


def minimum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.min(x1, x2, out=out)


minimum.support_native_out = True


def maximum(x1, x2, *, out: Optional[torch.Tensor] = None):
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.max(x1, x2, out=out)


maximum.support_native_out = True
