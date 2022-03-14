# global
import numpy as np
import numpy.array_api as npa

import typing

import ivy.functional.backends.numpy.core.general

def _cast_for_binary_op(x1: np.ndarray, x2: np.ndarray)\
        -> typing.Tuple[typing.Union[np.ndarray, int, float, bool], typing.Union[np.ndarray, int, float, bool]]:
    x1_bits = ivy.functional.backends.numpy.core.general.dtype_bits(x1.dtype)
    if isinstance(x2, (int, float, bool)):
        return x1, x2
    x2_bits = ivy.functional.backends.numpy.core.general.dtype_bits(x2.dtype)
    if x1_bits > x2_bits:
        x2 = x2.astype(x1.dtype)
    elif x2_bits > x1_bits:
        x1 = x1.astype(x2.dtype)
    return x1, x2

def bitwise_and(x1: np.ndarray,
                x2: np.ndarray)\
        -> np.ndarray:
    return np.bitwise_and(x1, x2)


def equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 == x2


def less_equal(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return x1 <= x2


def ceil(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.ceil(npa.asarray(x)))


def sqrt(x: np.ndarray)\
        -> np.ndarray:
    return np.sqrt(x)


def isfinite(x: np.ndarray) \
        -> np.ndarray:
    return np.asarray(npa.isfinite(npa.asarray(x)))


def asinh(x: np.ndarray)\
        -> np.ndarray:
    return np.arcsinh(x)


def cosh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.cosh(npa.asarray(x)))


def log10(x: np.ndarray)\
        -> np.ndarray:
    return np.log10(x)


def log2(x: np.ndarray)\
        -> np.ndarray:
    return np.log2(x)


def log1p(x: np.ndarray)\
        -> np.ndarray:
    return np.log1p(x)


def isnan(x: np.ndarray)\
        -> np.ndarray:
    return np.isnan(x)


def less(x1: np.ndarray,x2: np.ndarray)\
        -> np.ndarray:
    return np.less(x1,x2)


def cos(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.cos(npa.asarray(x)))


def logical_not(x: np.ndarray)\
        -> np.ndarray:
    return np.logical_not(x)


def logical_or(x1: np.ndarray, x2: np.ndarray)\
        -> np.ndarray:
    return np.logical_or(x1, x2)


def acosh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.acosh(npa.asarray(x)))
  

def sin(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.sin(npa.asarray(x)))

  
def negative(x: np.ndarray) -> np.ndarray:
    return np.negative(x)


def tanh(x: np.ndarray)\
        -> np.ndarray:
    return np.asarray(npa.tanh(npa.asarray(x)))


def bitwise_or(x1: np.ndarray , x2: np.ndarray) \
        -> np.ndarray:
    x1, x2 = _cast_for_binary_op(x1, x2)

    # numpy doesn't return an array that has same dtype as
    # x1 when doing bitwise_or with a scalar x2, therefore,
    # we manually need to convert scalar into np.ndarray of
    # same dtype as x1.
    #
    # however, this might be incorrect since when bitwise-or-ing
    # uint8 and int64, we convert int64 to uint8. This leads to loss in information.
    # NO idea why it is giving correct answers ? Even the answer in pytest is expected to be in uint8.
    # However, at the below link
    # https://data-apis.org/array-api/latest/API_specification/type_promotion.html#mixed-unsigned-and-signed-integer-type-promotion-table
    # it says the actual answer is to be expected in int64
    if not isinstance(x2, np.ndarray):
        x2 = np.asarray(x2, dtype=x1.dtype)

    return np.bitwise_or(x1, x2)
