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
    if isinstance(x1, torch.Tensor):
        if isinstance(x2, torch.Tensor):
            promoted_type = torch.promote_types(x1.dtype, x2.dtype)
            if clamp:
                x2 = torch.clamp(x2, max=torch.iinfo(promoted_type).bits - 1)
            x1 = x1.to(promoted_type)
            x2 = x2.to(promoted_type)
        else:
            x2 = torch.tensor(x2, dtype=x1.dtype)
    else:
        if isinstance(x2, torch.Tensor):
            x1 = torch.tensor(x1, dtype=x2.dtype)
        else:
            x1 = torch.tensor(x1)
            x2 = torch.tensor(x2)
    return x1, x2


def add(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.add(x1, x2, out=out)


def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_xor(x1, x2, out=out)


def expm1(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.expm1(x, out=out)


expm1.unsupported_dtypes = ("float16",)


def bitwise_invert(
    x: Union[int, bool, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


def isfinite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)


def isinf(x: torch.Tensor) -> torch.Tensor:
    return torch.isinf(x)


def equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.eq(x1, x2, out=out)


def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.less_equal(x1, x2, out=out)


def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_and(x1, x2, out=out)


def ceil(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.unsupported_dtypes = ("float16",)


def floor(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.unsupported_dtypes = ("float16",)


def asin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.asin(x, out=out)


asin.unsupported_dtypes = ("float16",)


def asinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.asinh(x, out=out)


asinh.unsupported_dtypes = ("float16",)


def sign(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sign(x, out=out)


def sqrt(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sqrt(x, out=out)


sqrt.unsupported_dtypes = ("float16",)


def cosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.cosh(x, out=out)


cosh.unsupported_dtypes = ("float16",)


def log10(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log10(x, out=out)


log10.unsupported_dtypes = ("float16",)


def log2(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log2(x, out=out)


log2.unsupported_dtypes = ("float16",)


def log1p(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log1p(x, out=out)


log1p.unsupported_dtypes = ("float16",)


def isnan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    ret = torch.isnan(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.lt(x1, x2, out=out)


def multiply(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.multiply(x1, x2, out=out)


def cos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.cos(x, out=out)


cos.unsupported_dtypes = ("float16",)


def logical_not(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool), out=out)


def divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.div(x1, x2, out=out)


def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.greater(x1, x2, out=out)


def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


def acos(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.acos(x, out=out)


acos.unsupported_dtypes = ("float16",)


def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


def acosh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.acosh(x, out=out)


acosh.unsupported_dtypes = ("float16",)


def sin(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sin(x, out=out)


sin.unsupported_dtypes = ("float16",)


def negative(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.neg(x, out=out)


def not_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.not_equal(x1, x2, out=out)


def tanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tanh(x, out=out)


tanh.unsupported_dtypes = ("float16",)


def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.div(x1, x2, rounding_mode="floor", out=out)


def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_or(x1, x2, out=out)


def sinh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.sinh(x, out=out)


sinh.unsupported_dtypes = ("float16",)


def positive(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    ret = torch.positive(x)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def square(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.square(x, out=out)


def pow(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.pow(x1, x2, out=out)


def round(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.round(x, out=out)


round.unsupported_dtypes = ("float16",)


def trunc(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if "int" not in str(x.dtype):
        return torch.trunc(x, out=out)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.unsupported_dtypes = ("float16",)


def abs(
    x: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.abs(x, out=out)


def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.unsupported_dtypes = ("float16",)


def tan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tan(x, out=out)


tan.unsupported_dtypes = ("float16",)


def atan(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.atan(x, out=out)


atan.unsupported_dtypes = ("float16",)


def atan2(
    x1: torch.Tensor, x2: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.unsupported_dtypes = ("float16",)


def log(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.log(x, out=out)


log.unsupported_dtypes = ("float16",)


def exp(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.exp(x, out=out)


exp.unsupported_dtypes = ("float16",)


def subtract(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.subtract(x1, x2, out=out)


def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = torch.remainder(x1, x2, out=out)
    ret[torch.isnan(ret)] = 0
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def atanh(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.atanh(x, out=out)


atanh.unsupported_dtypes = ("float16",)


def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2, clamp=True)
    return torch.bitwise_right_shift(x1, x2, out=out)


def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2, clamp=True)
    return torch.bitwise_left_shift(x1, x2, out=out)


# Extra #
# ------#


def erf(x: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.erf(x, out=out)


def minimum(x1, x2, *, out: Optional[torch.Tensor] = None):
    x_val = torch.tensor(x1) if (isinstance(x1, int) or isinstance(x1, float)) else x1
    y_val = torch.tensor(x2) if (isinstance(x2, int) or isinstance(x2, float)) else x2
    return torch.min(x_val, y_val, out=out)


def maximum(x1, x2, *, out: Optional[torch.Tensor] = None):
    x_val = torch.tensor(x1) if (isinstance(x1, int) or isinstance(x1, float)) else x1
    y_val = torch.tensor(x2) if (isinstance(x2, int) or isinstance(x2, float)) else x2
    return torch.max(x_val, y_val, out=out)
