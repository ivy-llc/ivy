# global
from typing import Union, Optional
from math import pi
import torch

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    handle_numpy_arrays_in_specific_backend,
)
from ivy import promote_types_of_inputs
from . import backend_version


def _cast_for_unary_op(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x


@handle_numpy_arrays_in_specific_backend
def add(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.add(x1, x2, alpha=alpha, out=out)
    return torch.add(x1, x2, out=out)


add.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


def imag(
    val: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if val.dtype not in (torch.complex64, torch.complex128):
        ret = torch.imag(val.to(torch.complex64))
        return ret.to(val.dtype)
    return torch.imag(val)


imag.support_native_out = False


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def expm1(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.expm1(x, out=out)


expm1.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_invert(
    x: Union[int, bool, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


bitwise_invert.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isfinite(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.isfinite(x)


@handle_numpy_arrays_in_specific_backend
def isinf(
    x: torch.Tensor,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if detect_negative and detect_positive:
        return torch.isinf(x)
    elif detect_negative:
        return torch.isneginf(x)
    elif detect_positive:
        return torch.isposinf(x)
    return torch.full_like(x, False, dtype=torch.bool)


@handle_numpy_arrays_in_specific_backend
def equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.eq(x1, x2, out=out)


equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def ceil(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def fmin(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fmin(x1, x2, out=None)


fmin.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.asin(x, out=out)


asin.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def asinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.asinh(x, out=out)


asinh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sign(
    x: torch.Tensor,
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "complex" in str(x.dtype):
        if np_variant:
            return torch.where(
                x.real != 0, torch.sign(x.real) + 0.0j, torch.sign(x.imag) + 0.0j
            )
        return torch.sgn(x, out=out)
    return torch.sign(x, out=out)


sign.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sqrt(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sqrt(x, out=out)


sqrt.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.cosh(x, out=out)


cosh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log10(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log10(x, out=out)


log10.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log2(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log2(x, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log1p(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log1p(x, out=out)


log1p.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def isnan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.isnan(x)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.lt(x1, x2, out=out)


less.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def multiply(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.multiply(x1, x2, out=out)


multiply.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def cos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.cos(x, out=out)


cos.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_not(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.logical_not(x.type(torch.bool), out=out)


logical_not.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2)
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
        ret = ivy.astype(ret, x1.dtype, copy=False)
    else:
        ret = ivy.astype(ret, ivy.default_float_dtype(as_native=True), copy=False)
    return ret


divide.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater(x1, x2, out=out)


greater.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.acos(x, out=out)


acos.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def lcm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.lcm(x1, x2, out=out)


lcm.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_xor.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_and.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_or.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def acosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.acosh(x, out=out)


acosh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sin(x, out=out)


sin.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def negative(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.neg(x, out=out)


negative.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def not_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.tanh(x, out=out)


tanh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if ivy.exists(out):
        if not ivy.is_float_dtype(out):
            return ivy.inplace_update(
                out, torch.floor(torch.div(x1, x2)).type(out.dtype)
            )
    return torch.floor(torch.div(x1, x2), out=out).type(x1.dtype)


floor_divide.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def sinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinh(x, out=out)


sinh.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def positive(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.positive(x)


@handle_numpy_arrays_in_specific_backend
def square(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.square(x, out=out)


square.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def pow(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.pow(x1, x2, out=out)


pow.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def round(
    x: torch.Tensor, /, *, decimals: int = 0, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.round(x, decimals=decimals, out=out)


round.support_native_out = True


def trapz(
    y: torch.Tensor,
    /,
    *,
    x: Optional[torch.Tensor] = None,
    dx: Optional[float] = None,
    axis: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x is None:
        dx = dx if dx is not None else 1
        return torch.trapezoid(y, dx=dx, dim=axis)
    else:
        if dx is not None:
            TypeError(
                "trapezoid() received an invalid combination of arguments - got "
                "(Tensor, Tensor, int), but expected one of: *(Tensor "
                "y, Tensor x, *, int dim) * (Tensor y, *, Number dx, int dim)"
            )
        else:
            return torch.trapezoid(y, x=x, dim=axis)


trapz.support_native_out = False


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" not in str(x.dtype):
        return torch.trunc(x, out=out)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def abs(
    x: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if x.dtype is torch.bool:
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.abs(x, out=out)


abs.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def logaddexp2(
    x1: Union[torch.Tensor, float, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.type(ivy.default_float_dtype(as_native=True))
        x2 = x2.type(ivy.default_float_dtype(as_native=True))
    return torch.logaddexp2(x1, x2, out=out)


logaddexp2.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.tan(x, out=out)


tan.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.atan(x, out=out)


atan.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("float16", "bfloat16", "complex")}, backend_version
)  # TODO Fixed in PyTorch 1.12.1 (this note excludes complex)
@handle_numpy_arrays_in_specific_backend
def atan2(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def log(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log(x, out=out)


log.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def exp(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.exp(x, out=out)


exp.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def exp2(
    x: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.exp2(x, out=out)


exp2.support_native_out = True


@handle_numpy_arrays_in_specific_backend
def subtract(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        return torch.subtract(x1, x2, alpha=alpha, out=out)
    return torch.subtract(x1, x2, out=out)


subtract.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = torch.where(res >= 0, torch.floor(res), torch.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        if ivy.exists(out):
            if out.dtype != x2.dtype:
                return ivy.inplace_update(
                    out, torch.round(torch.mul(diff, x2)).to(out.dtype)
                )
        return torch.round(torch.mul(diff, x2), out=out).to(x1.dtype)
    return torch.remainder(x1, x2, out=out).to(x1.dtype)


remainder.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def atanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.atanh(x, out=out)


atanh.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    x2 = torch.clamp(x2, min=0, max=torch.iinfo(x2.dtype).bits - 1)
    return torch.bitwise_right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return torch.bitwise_left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, backend_version)
@handle_numpy_arrays_in_specific_backend
def erf(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.erf(x, out=out)


erf.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def minimum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return torch.where(x1 <= x2, x1, x2, out=out)
    return torch.minimum(x1, x2, out=out)


minimum.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def maximum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    use_where: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return torch.where(x1 >= x2, x1, x2, out=out)
    return torch.maximum(x1, x2, out=out)


maximum.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def reciprocal(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.reciprocal(x, out=out)


reciprocal.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def deg2rad(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.deg2rad(x, out=out)


deg2rad.support_native_out = True


@with_unsupported_dtypes(
    {"2.0.1 and below": ("complex64", "complex128")}, backend_version
)
@handle_numpy_arrays_in_specific_backend
def rad2deg(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.rad2deg(x, out=out)


rad2deg.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
@handle_numpy_arrays_in_specific_backend
def trunc_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2, rounding_mode="trunc")
    if ivy.is_float_dtype(x1.dtype):
        ret = ret.to(x1.dtype)
    else:
        ret = ret.to(ivy.default_float_dtype(as_native=True))
    return ret


@handle_numpy_arrays_in_specific_backend
def isreal(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.isreal(x)


@with_unsupported_dtypes(
    {"2.0.1 and below": ("bfloat16", "complex")},
    backend_version,
)
@handle_numpy_arrays_in_specific_backend
def fmod(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.fmod(x1, x2, out=None)


fmod.support_native_out = True


def gcd(
    x1: Union[torch.Tensor, int, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.gcd(x1, x2, out=out)


gcd.support_native_out = True


def angle(
    input: torch.Tensor,
    /,
    *,
    deg: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if deg:
        return torch.angle(input, out=out) * (180 / pi)
    else:
        return torch.angle(input, out=out)


angle.support_native_out = True


def nan_to_num(
    x: torch.Tensor,
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if copy:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=out)
    else:
        x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
        return x


def real(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.real(x)
