# global
import torch
from torch import Tensor
import typing


# local
import ivy


def isfinite(x: Tensor)\
        -> Tensor:
    return torch.isfinite(x)


def _cast_for_binary_op(x1: Tensor, x2: Tensor)\
        -> typing.Tuple[typing.Union[Tensor, int, float, bool], typing.Union[Tensor, int, float, bool]]:
    x1_bits = ivy.functional.backends.torch.core.general.dtype_bits(x1.dtype)
    if isinstance(x2, (int, float, bool)):
        return x1, x2
    x2_bits = ivy.functional.backends.torch.core.general.dtype_bits(x2.dtype)
    if x1_bits > x2_bits:
        x2 = x2.type(x1.dtype)
    elif x2_bits > x1_bits:
        x1 = x1.type(x2.dtype)
    return x1, x2


def equal(x1: Tensor, x2: Tensor)\
        -> Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return x1 == x2


def less_equal(x1: Tensor, x2: Tensor)\
        -> Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return x1 <= x2


def bitwise_and(x1: torch.Tensor,
                x2: torch.Tensor)\
        -> torch.Tensor:
    return torch.bitwise_and(x1, x2)


def ceil(x: torch.Tensor)\
        -> torch.Tensor:
    if 'int' in str(x.dtype):
        return x
    return torch.ceil(x)


def isfinite(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.isfinite(x)


def asinh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.asinh(x)


def sqrt(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.sqrt(x)


def cosh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.cosh(x)


def log2(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.log2(x)


def log1p(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.log1p(x)


def isnan(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.isnan(x)


def less(x1: torch.Tensor, x2: torch.Tensor):
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.lt(x1, x2)


def cos(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.cos(x)


def logical_not(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool))
