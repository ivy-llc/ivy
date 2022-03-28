# global
import numpy as np
import numpy.array_api as npa

try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None


def expm1(x: np.ndarray)\
        -> np.ndarray:
    return np.expm1(x)


def bitwise_invert(x: np.ndarray)\
        -> np.ndarray:
    return np.invert(x)


def bitwise_and(x1: np.ndarray,
                x2: np.ndarray)\
        -> np.ndarray:
    return np.bitwise_and(x1, x2)


def equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 == x2


def greater_equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.greater_equal(x1, x2)


def less_equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 <= x2


def multiply(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, 'dtype'):
        x2 = np.array(x2, dtype=x1.dtype)
    return np.multiply(x1, x2)


def ceil(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.ceil(npa.asarray(x)))


def floor(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.floor(npa.asarray(x)))


def sqrt(x: np.ndarray)\
        -> np.ndarray:
    return np.sqrt(x)


def isfinite(x: np.ndarray) \
        -> np.ndarray:
    return np.asarray(npa.isfinite(npa.asarray(x)))


def asin(x: np.ndarray)\
        -> np.ndarray:
    return np.arcsin(x)


def isinf(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.isinf(npa.asarray(x)))


def asinh(x: np.ndarray)\
        -> np.ndarray:
    return np.arcsinh(x)


def cosh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.cosh(npa.asarray(x)))


def log10(x: np.ndarray)\
        -> np.ndarray:
    return np.log10(x)


def log(x: np.ndarray)\
        -> np.ndarray:
    return np.log(x)


def log2(x: np.ndarray)\
        -> np.ndarray:
    return np.log2(x)


def log1p(x: np.ndarray)\
        -> np.ndarray:
    return np.log1p(x)


def isnan(x: np.ndarray)\
        -> np.ndarray:
    return np.isnan(x)


def less(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.less(x1, x2)


def cos(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.cos(npa.asarray(x)))


def logical_not(x: np.ndarray)\
        -> np.ndarray:
    return np.logical_not(x)


def acos(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.acos(npa.asarray(x)))


def logical_xor(x1: np.ndarray, x2: np.ndarray) \
        -> np.ndarray:
    return np.logical_xor(x1, x2)


def logical_or(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.logical_or(x1, x2)


def logical_and(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.logical_and(x1, x2)


def acosh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.acosh(npa.asarray(x)))


def sin(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.sin(npa.asarray(x)))


def negative(x: np.ndarray) -> np.ndarray:
    return np.negative(x)


def not_equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.not_equal(x1, x2)


def tanh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.tanh(npa.asarray(x)))


def sinh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.sinh(npa.asarray(x)))


def positive(x: np.ndarray)\
        -> np.ndarray:
    return np.positive(x)


def square(x: np.ndarray)\
        -> np.ndarray:
    return np.square(x)


def remainder(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.remainder(x1, x2)


def round(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.round(npa.asarray(x)))


def bitwise_or(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.bitwise_or(x1, x2)


def abs(x: np.ndarray)\
        -> np.ndarray:
    return np.absolute(x)


def subtract(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    if hasattr(x1, 'dtype') and hasattr(x2, 'dtype'):
        promoted_type = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(promoted_type)
        x2 = x2.astype(promoted_type)
    elif not hasattr(x2, 'dtype'):
        x2 = np.array(x2, dtype=x1.dtype)
    return np.subtract(x1, x2)


def logaddexp(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)
    else:
        dtype = np.promote_types(x1.dtype, x2.dtype)
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    return np.logaddexp(x1, x2)


tan = np.tan


def atan(x: np.ndarray) \
        -> np.ndarray:
    return np.arctan(x)


atan2 = np.arctan2
cosh = np.cosh
atanh = np.arctanh
log = np.log
exp = np.exp


# Extra #
# ------#


def erf(x):
    if _erf is None:
        raise Exception('scipy must be installed in order to call ivy.erf with a numpy backend.')
    return _erf(x)
