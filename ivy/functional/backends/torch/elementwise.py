# global
import torch
from torch import Tensor
import typing
import math

# local
import ivy


def bitwise_xor(x1: torch.Tensor,
                x2: torch.Tensor)\
        -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_xor(x1, x2)


def expm1(x: Tensor)\
        -> Tensor:
    return torch.expm1(x)


def bitwise_invert(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.bitwise_not(x)


def isfinite(x: Tensor)\
        -> Tensor:
    return torch.isfinite(x)


def isinf(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.isinf(x)


def _cast_for_binary_op(x1: Tensor, x2: Tensor)\
        -> typing.Tuple[typing.Union[Tensor, int, float, bool], typing.Union[Tensor, int, float, bool]]:
    x1_bits = ivy.functional.backends.torch.dtype_bits(x1.dtype)
    if isinstance(x2, (int, float, bool)):
        return x1, x2
    x2_bits = ivy.functional.backends.torch.dtype_bits(x2.dtype)
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


def floor(x: torch.Tensor)\
        -> torch.Tensor:
    if 'int' in str(x.dtype):
        return x
    return torch.floor(x)


def isfinite(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.isfinite(x)


def asin(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.asin(x)
  

def asinh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.asinh(x)


def sqrt(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.sqrt(x)


def cosh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.cosh(x)


def log10(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.log10(x)


def log(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.log(x)


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


def multiply(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.multiply(x1, x2)


def cos(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.cos(x)


def logical_not(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.logical_not(x.type(torch.bool))


def greater_equal(x1: torch.Tensor, x2: torch.Tensor):
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.greater_equal(x1, x2)


def acos(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.acos(x)


def logical_xor(x1: torch.Tensor, x2: torch.Tensor) \
        -> torch.Tensor:
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool))


def logical_and(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool))

  
def logical_or(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool))


def acosh(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.acosh(x)

  
def sin(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.sin(x)


def negative(x: torch.Tensor) -> torch.Tensor:
    return torch.neg(x)


def not_equal(x1: Tensor, x2: Tensor)\
        -> Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.not_equal(x1, x2)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def bitwise_or(x1: torch.Tensor, x2: torch.Tensor) \
        -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.bitwise_or(x1, x2)


def sinh(x: torch.Tensor) -> torch.Tensor:
    return torch.sinh(x)


def positive(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.positive(x)

    
def square(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.square(x)


def round(x: torch.Tensor)\
        -> torch.Tensor:
    if 'int' in str(x.dtype):
        return x
    return torch.round(x)


def abs(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.abs(x)

  
def logaddexp(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.logaddexp(x1, x2)


def tan(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.tan(x)


def acos(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.acos(x)


def atan(x: torch.Tensor) \
        -> torch.Tensor:
    return torch.atan(x)


def atan2(x1: Tensor, x2: Tensor) -> Tensor:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.atan2(x1, x2)


def cosh(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.cosh(x)



def log(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.log(x)


def exp(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.exp(x)


def subtract(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.subtract(x1, x2)


def remainder(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return torch.remainder(x1, x2)



def atanh(x: torch.Tensor) \
        -> torch.Tensor:
    if isinstance(x, float):
        return math.atanh(x)
    return torch.atanh(x)



def bitwise_right_shift(x1: torch.Tensor, x2: torch.Tensor)\
        -> torch.Tensor:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = torch.promote_types(x1.dtype, x2.dtype)
        x2 = torch.clamp(x2, max=torch.iinfo(promoted_type).bits - 1)
        x1 = x1.to(promoted_type)
        x2 = x2.to(promoted_type)
    return torch.bitwise_right_shift(x1, x2)


# Extra #
# ------#


def erf(x: torch.Tensor)\
        -> torch.Tensor:
    return torch.erf(x)


def minimum(x, y):
    x_val = torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return torch.min(x_val, y_val)


def maximum(x, y):
    x_val = torch.tensor(x) if (isinstance(x, int) or isinstance(x, float)) else x
    y_val = torch.tensor(y) if (isinstance(y, int) or isinstance(y, float)) else y
    return torch.max(x_val, y_val)
