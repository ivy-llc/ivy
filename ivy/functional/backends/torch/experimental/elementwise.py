# global
from typing import Optional, Union, Tuple, List
from numbers import Number
import torch

# local
import ivy
from ivy import promote_types_of_inputs
from ivy.functional.backends.torch.elementwise import _cast_for_unary_op
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
)
from .. import backend_version


@with_supported_dtypes({"2.0.1 and below": ("float32", "float64")}, backend_version)
def lgamma(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.lgamma(x, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def fmax(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.fmax(x1, x2, out=None)


fmax.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def sinc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinc(x, out=out)


sinc.support_native_out = True


def float_power(
    x1: Union[torch.Tensor, float, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Native out is supported but with restrictions leading
    # to failures hence letting ivy handle it.
    x1, x2 = promote_types_of_inputs(x1, x2)
    return torch.float_power(x1, x2, out=out)


float_power.support_native_out = True


def copysign(
    x1: Union[torch.Tensor, Number],
    x2: Union[torch.Tensor, Number],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.type(ivy.default_float_dtype(as_native=True))
        x2 = x2.type(ivy.default_float_dtype(as_native=True))
    return torch.copysign(torch.as_tensor(x1), x2, out=out)


copysign.support_native_out = True


def count_nonzero(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    if dtype is None:
        x = torch.count_nonzero(a, dim=axis)
    else:
        x = torch.tensor(torch.count_nonzero(a, dim=axis), dtype=dtype)
    if not keepdims:
        return x
    if isinstance(axis, int):
        if axis == -1:
            temp = x.dim() - 2
            if temp < -1:
                temp = 0
            return x.unsqueeze(temp)
        return x.unsqueeze(axis - 1)
    elif axis is not None:
        for d in sorted(axis):
            x = x.unsqueeze(d - 1)
        return x
    return x


count_nonzero.support_native_out = False


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, backend_version)
def nansum(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int, ...], int]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    return torch.nansum(x, dim=axis, keepdim=keepdims, dtype=dtype)


nansum.support_native_out = False


def isclose(
    a: torch.Tensor,
    b: torch.Tensor,
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


isclose.support_native_out = False


def diff(
    x: Union[torch.Tensor, list, tuple],
    /,
    *,
    n: int = 1,
    axis: int = -1,
    prepend: Optional[Union[torch.Tensor, int, float, list, tuple]] = None,
    append: Optional[Union[torch.Tensor, int, float, list, tuple]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = x if type(x) == torch.Tensor else torch.tensor(x)
    prepend = (
        prepend
        if type(prepend) == torch.Tensor or prepend is None
        else torch.tensor(prepend)
    )
    append = (
        append
        if type(append) == torch.Tensor or append is None
        else torch.tensor(append)
    )
    return torch.diff(x, n=n, dim=axis, prepend=prepend, append=append)


def signbit(
    x: Union[torch.Tensor, float, int, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.signbit(x, out=out)


signbit.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def hypot(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.hypot(x1, x2)


def allclose(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    out: Optional[torch.Tensor] = None,
) -> bool:
    ret = torch.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return torch.tensor(ret)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def fix(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fix(x, out=out)


fix.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def nextafter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nextafter(x1, x2)


nextafter.support_native_out = True


def zeta(
    x: torch.Tensor,
    q: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    temp = torch.logical_and(torch.ne(torch.remainder(x, 2), 0), torch.gt(x, 1))
    temp = torch.logical_and(temp, torch.le(q, 0))
    nan_indices = torch.logical_or(temp, torch.lt(x, 1))
    result = torch.special.zeta(x, q)
    result.masked_fill_(nan_indices, float("nan"))
    return result


zeta.support_native_out = False


def gradient(
    x: torch.Tensor,
    /,
    *,
    spacing: Union[int, list, tuple] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: int = 1,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if axis is None:
        axis = tuple(range(len(x.shape)))
    if type(axis) == int:
        axis = (axis,)
    if type(spacing) == int:
        spacing = [spacing] * len(axis)

    grad = torch.gradient(x, spacing=spacing, dim=axis, edge_order=edge_order)
    if len(grad) == 1:
        return grad[0]
    return grad


@with_supported_dtypes(
    {"2.0.1 and below": ("float16", "float32", "float64")},
    backend_version,
)
def xlogy(
    x: torch.tensor, y: torch.tensor, /, *, out: Optional[torch.tensor] = None
) -> torch.tensor:
    x, y = promote_types_of_inputs(x, y)
    return torch.xlogy(x, y, out=out)


def conj(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    conj_x = torch.conj(x)
    return torch.resolve_conj(input=conj_x)


def ldexp(
    x1: torch.Tensor,
    x2: Union[int, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ldexp(x1, x2, out=out)


def _are_suitable_types_for_torch_lerp(input, end, weight):
    suitable_types = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]

    if not isinstance(input, torch.Tensor) or not isinstance(end, torch.Tensor):
        return False
    else:
        if input.dtype not in suitable_types or end.dtype not in suitable_types:
            return False

    if not isinstance(weight, float) and not isinstance(weight, torch.Tensor):
        return False
    else:
        if isinstance(weight, torch.Tensor):
            if weight.dtype not in [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ]:
                return False

    return True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def lerp(
    input: torch.Tensor,
    end: torch.Tensor,
    weight: Union[torch.Tensor, float],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.lerp(input, end, weight, out=out)


lerp.partial_mixed_handler = lambda input, end, weight, **kwargs: (
    _are_suitable_types_for_torch_lerp(input, end, weight)
)
lerp.support_native_out = True


def frexp(
    x: torch.Tensor,
    /,
    *,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mantissa, exponent = torch.frexp(x, out=out)
    return mantissa, exponent


def modf(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    modf_x = torch.modf(x)
    return torch.resolve_modf(input=modf_x)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, backend_version)
def digamma(
    x: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.special.digamma(x, out=out)


digamma.support_native_out = True


def amax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, list, tuple]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    amax_x = torch.reduce_max(x, axis=axis, keepdims=keepdims)
    return torch.resolve_max(input=amax_x, out=out)
