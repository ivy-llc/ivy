# global
from typing import Optional, Union
import torch

# local
from ivy.functional.backends.torch.elementwise import _cast_for_unary_op
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


def lcm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.abs(torch.lcm(x1, x2, out=out))


lcm.support_native_out = True


def fmod(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fmod(x1, x2, out=None)


fmod.support_native_out = True
fmod.unsupported_dtypes = ("bfloat16",)


def fmax(
    x1: torch.Tensor,
    x2: torch.Tensor,
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.fmax(x1, x2, out=None)


fmax.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def sinc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinc(x, out=out)


sinc.support_native_out = True


def trapz(
    y: torch.Tensor,
    /,
    *,
    x: Optional[torch.Tensor] = None,
    dx: Optional[float] = None,
    axis: Optional[int] = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x is None:
        dx = dx if dx is not None else 1
        return torch.trapezoid(y, dx=dx, dim=axis)
    else:
        if dx is not None:
            TypeError(
                "trapezoid() received an invalid combination of arguments - got\
            (Tensor, Tensor, int), but expected one of: *\
            (Tensor y, Tensor x, *, int dim) * (Tensor y, *, Number dx, int dim)"
            )
        else:
            return torch.trapezoid(y, x=x, dim=axis)


trapz.support_native_out = False


def float_power(
    x1: Union[torch.Tensor, float, list, tuple],
    x2: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.tensor(torch.float_power(x1, x2, out=out), dtype=x1.dtype)


float_power.support_native_out = True


def exp2(
    x: Union[torch.Tensor, float, list, tuple],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.exp2(x, out=out)


exp2.support_native_out = True
