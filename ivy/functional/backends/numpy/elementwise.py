# global
import numpy as np
from typing import Union, Optional


# local
import ivy

try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None


def add(x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]) -> np.ndarray:
    if not isinstance(x1, np.ndarray):
        x1 = np.asarray(x1)
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2)
    return np.asarray(np.add(x1, x2))


def pow(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = np.promote_types(x1.dtype, x2.dtype)
        x1, x2 = np.asarray(x1), np.asarray(x2)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = np.array(x2, dtype=x1.dtype)
    return np.power(x1, x2, out=out)


def bitwise_xor(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.bitwise_xor(x1, x2, out=out)


def exp(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(x, out=out)


def expm1(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.expm1(x, out=out)


def bitwise_invert(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.invert(x, out=out)


def bitwise_and(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.bitwise_and(x1, x2, out=out)


def equal(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.equal(x1, x2, out=out)


def greater(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater(x1, x2, out=out)


def greater_equal(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater_equal(x1, x2, out=out)


def less_equal(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less_equal(x1, x2, out=out)


def multiply(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = np.promote_types(x1.dtype, x2.dtype)
        x1, x2 = np.asarray(x1), np.asarray(x2)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = np.array(x2, dtype=x1.dtype)
    return np.multiply(x1, x2, out=out)


def ceil(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def floor(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def sign(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sign(x, out=out)


def sqrt(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sqrt(x, out=out)


def isfinite(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isfinite(x, out=out)


def asin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsin(x, out=out)


def isinf(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isinf(x, out=out)


def asinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsinh(x, out=out)


def cosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cosh(x, out=out)


def log10(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log10(x, out=out)


def log(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log(x, out=out)


def log2(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log2(x, out=out)


def log1p(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log1p(x, out=out)


def isnan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isnan(x, out=out)


def less(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less(x1, x2, out=out)


def cos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cos(x, out=out)


def logical_not(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.logical_not(x, out=out)


def divide(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if isinstance(x1, np.ndarray):
        if not isinstance(x2, np.ndarray):
            x2 = np.asarray(x2, dtype=x1.dtype)
        else:
            promoted_type = np.promote_types(x1.dtype, x2.dtype)
            x1 = x1.astype(promoted_type)
            x2 = x2.astype(promoted_type)
    return np.divide(x1, x2, out=out)


def acos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccos(x, out=out)


def logical_xor(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_xor(x1, x2, out=out)


def logical_or(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_or(x1, x2, out=out)


def logical_and(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_and(x1, x2, out=out)


def acosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccosh(x, out=out)


def sin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sin(x, out=out)


def negative(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.negative(x, out=out)


def not_equal(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.not_equal(x1, x2, out=out)


def tanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tanh(x, out=out)


def floor_divide(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.floor_divide(x1, x2, out=out)


def sinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinh(x, out=out)


def positive(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.positive(x, out=out)


def square(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.square(x, out=out)


def remainder(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.remainder(x1, x2, out=out)


def round(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.round(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def bitwise_or(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.bitwise_or(x1, x2, out=out)


def trunc(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.trunc(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def abs(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.absolute(x, out=out)


def subtract(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if hasattr(x1, "dtype") and hasattr(x2, "dtype"):
        promoted_type = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, "dtype"):
        x2 = np.array(x2, dtype=x1.dtype)
    return np.subtract(x1, x2, out=out)


def logaddexp(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.logaddexp(x1, x2, out=out)


def bitwise_right_shift(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.right_shift(x1, x2, out=out)


def bitwise_left_shift(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.left_shift(x1, x2, out=out)


def tan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tan(x, out=out)


def atan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctan(x, out=out)


def atanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctanh(x, out=out)


def atan2(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.arctan2(x1, x2, out=out)


# Extra #
# ------#


def minimum(x1, x2, *, out: Optional[np.ndarray] = None):
    return np.minimum(x1, x2, out=out)


def maximum(x1, x2, *, out: Optional[np.ndarray] = None):
    return np.maximum(x1, x2, out=out)


def erf(x, *, out: Optional[np.ndarray] = None):
    if _erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.erf with a numpy backend."
        )
    return _erf(x, out=out)
