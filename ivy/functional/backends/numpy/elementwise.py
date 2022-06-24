# global
import numpy as np
from typing import Union, Optional, Callable
import functools

# local
import ivy

try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None


def _cast_for_binary_op(x1, x2):
    if isinstance(x1, np.ndarray):
        if isinstance(x2, np.ndarray):
            promoted_type = np.promote_types(x1.dtype, x2.dtype)
            x1 = x1.astype(promoted_type)
            x2 = x2.astype(promoted_type)
        else:
            x2 = np.asarray(x2, dtype=x1.dtype)
    return x1, x2


# when inputs are 0 dimensional, numpy's functions return scalars
# so we use this wrapper to ensure outputs are always numpy arrays
def _handle_0_dim_output(function: Callable) -> Callable:
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return np.asarray(ret) if not isinstance(ret, np.ndarray) else ret

    return new_function


@_handle_0_dim_output
def add(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.add(x1, x2, out=out)


@_handle_0_dim_output
def pow(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.power(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_xor(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.bitwise_xor(x1, x2, out=out)


@_handle_0_dim_output
def exp(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(x, out=out)


@_handle_0_dim_output
def expm1(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.expm1(x, out=out)


@_handle_0_dim_output
def bitwise_invert(
    x: Union[int, bool, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.invert(x, out=out)


@_handle_0_dim_output
def bitwise_and(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.bitwise_and(x1, x2, out=out)


@_handle_0_dim_output
def equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.equal(x1, x2, out=out)


@_handle_0_dim_output
def greater(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater(x1, x2, out=out)


@_handle_0_dim_output
def greater_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater_equal(x1, x2, out=out)


@_handle_0_dim_output
def less_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less_equal(x1, x2, out=out)


@_handle_0_dim_output
def multiply(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.multiply(x1, x2, out=out)


@_handle_0_dim_output
def ceil(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def floor(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def sign(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sign(x, out=out)


@_handle_0_dim_output
def sqrt(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sqrt(x, out=out)


@_handle_0_dim_output
def isfinite(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isfinite(x, out=out)


@_handle_0_dim_output
def asin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsin(x, out=out)


@_handle_0_dim_output
def isinf(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isinf(x, out=out)


@_handle_0_dim_output
def asinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsinh(x, out=out)


@_handle_0_dim_output
def cosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cosh(x, out=out)


@_handle_0_dim_output
def log10(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log10(x, out=out)


@_handle_0_dim_output
def log(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log(x, out=out)


@_handle_0_dim_output
def log2(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log2(x, out=out)


@_handle_0_dim_output
def log1p(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log1p(x, out=out)


@_handle_0_dim_output
def isnan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isnan(x, out=out)


@_handle_0_dim_output
def less(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less(x1, x2, out=out)


@_handle_0_dim_output
def cos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cos(x, out=out)


@_handle_0_dim_output
def logical_not(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.logical_not(x, out=out)


@_handle_0_dim_output
def divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.divide(x1, x2, out=out)


@_handle_0_dim_output
def acos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccos(x, out=out)


@_handle_0_dim_output
def logical_xor(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_xor(x1, x2, out=out)


@_handle_0_dim_output
def logical_or(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_or(x1, x2, out=out)


@_handle_0_dim_output
def logical_and(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_and(x1, x2, out=out)


@_handle_0_dim_output
def acosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccosh(x, out=out)


@_handle_0_dim_output
def sin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sin(x, out=out)


@_handle_0_dim_output
def negative(
    x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.negative(x, out=out)


@_handle_0_dim_output
def not_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.not_equal(x1, x2, out=out)


@_handle_0_dim_output
def tanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tanh(x, out=out)


@_handle_0_dim_output
def floor_divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    ret = np.floor_divide(x1, x2, out=out)
    # TODO: implement two more corner cases
    # 1) If `x1_i` is a positive (i.e., greater than `0`) finite number
    #    and `x2_i` is `-infinity`, the result is `-0`.
    # 2) If `x1_i` is a negative (i.e., less than `0`) finite number
    #    and `x2_i` is `+infinity`, the result is `-0`.
    if isinf(x1).any() and isfinite(x2).any():
        return ivy.full_like(ret, np.floor(np.divide(x1, x2)), dtype=ret.dtype)
    return ret


@_handle_0_dim_output
def sinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinh(x, out=out)


@_handle_0_dim_output
def positive(
    x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.positive(x, out=out)


@_handle_0_dim_output
def square(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.square(x, out=out)


@_handle_0_dim_output
def remainder(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.remainder(x1, x2, out=out)


@_handle_0_dim_output
def round(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.round(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def bitwise_or(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.bitwise_or(x1, x2, out=out)


@_handle_0_dim_output
def trunc(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.trunc(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


@_handle_0_dim_output
def abs(x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.absolute(x, out=out)


@_handle_0_dim_output
def subtract(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.subtract(x1, x2, out=out)


@_handle_0_dim_output
def logaddexp(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.logaddexp(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_right_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.right_shift(x1, x2, out=out)


@_handle_0_dim_output
def bitwise_left_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.left_shift(x1, x2, out=out)


@_handle_0_dim_output
def tan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tan(x, out=out)


@_handle_0_dim_output
def atan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctan(x, out=out)


@_handle_0_dim_output
def atanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctanh(x, out=out)


@_handle_0_dim_output
def atan2(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)
    return np.arctan2(x1, x2, out=out)


# Extra #
# ------#


@_handle_0_dim_output
def minimum(x1, x2, *, out: Optional[np.ndarray] = None):
    return np.minimum(x1, x2, out=out)


@_handle_0_dim_output
def maximum(x1, x2, *, out: Optional[np.ndarray] = None):
    return np.maximum(x1, x2, out=out)


@_handle_0_dim_output
def erf(x, *, out: Optional[np.ndarray] = None):
    if _erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.erf with a numpy backend."
        )
    return _erf(x, out=out)
