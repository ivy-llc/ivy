# global
from typing import Union, Optional

import numpy as np

# local
import ivy
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output

try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None


def _clamp_bits(x1, x2):
    x2 = np.clip(
        x2,
        np.array(0, dtype=x2.dtype),
        np.array(np.dtype(x1.dtype).itemsize * 8 - 1),
        dtype=x2.dtype,
    )
    return x1, x2


# when inputs are 0 dimensional, numpy's functions return scalars
# so we use this wrapper to ensure outputs are always numpy arrays


@_handle_0_dim_output
def abs(x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.absolute(x, out=out)


abs.support_native_out = True


@_handle_0_dim_output
def acos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccos(x, out=out)


acos.support_native_out = True


@_handle_0_dim_output
def acosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccosh(x, out=out)


acosh.support_native_out = True


@_handle_0_dim_output
def add(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.add(x1, x2, out=out)


add.support_native_out = True


@_handle_0_dim_output
def asin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsin(x, out=out)


asin.support_native_out = True


@_handle_0_dim_output
def asinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsinh(x, out=out)


asinh.support_native_out = True


@_handle_0_dim_output
def atan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctan(x, out=out)


atan.support_native_out = True


@_handle_0_dim_output
def atan2(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.arctan2(x1, x2, out=out)


atan2.support_native_out = True


@_handle_0_dim_output
def atanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctanh(x, out=out)


atanh.support_native_out = True


@_handle_0_dim_output
def bitwise_and(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@_handle_0_dim_output
def bitwise_invert(
    x: Union[int, bool, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.invert(x, out=out)


bitwise_invert.support_native_out = True


@_handle_0_dim_output
def bitwise_left_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = _clamp_bits(x1, x2)
    return np.left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


@_handle_0_dim_output
def bitwise_or(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@_handle_0_dim_output
def bitwise_right_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    x1, x2 = _clamp_bits(x1, x2)
    return np.right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@_handle_0_dim_output
def bitwise_xor(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


@_handle_0_dim_output
def ceil(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


ceil.support_native_out = True


@_handle_0_dim_output
def cos(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cos(x, out=out)


cos.support_native_out = True


@_handle_0_dim_output
def cosh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cosh(x, out=out)


cosh.support_native_out = True


@_handle_0_dim_output
def divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = np.divide(x1, x2)
    if ivy.is_float_dtype(x1):
        ret = np.asarray(ret, dtype=x1.dtype)
    else:
        ret = np.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


divide.support_native_out = True


@_handle_0_dim_output
def equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.equal(x1, x2, out=out)


equal.support_native_out = True


@_handle_0_dim_output
def exp(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(x, out=out)


exp.support_native_out = True


@_handle_0_dim_output
def expm1(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.expm1(x, out=out)


expm1.support_native_out = True


@_handle_0_dim_output
def floor(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


floor.support_native_out = True


@_handle_0_dim_output
def floor_divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = np.floor_divide(x1, x2, out=out)

    return ret


floor_divide.support_native_out = True


@_handle_0_dim_output
def greater(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater(x1, x2, out=out)


greater.support_native_out = True


@_handle_0_dim_output
def greater_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@_handle_0_dim_output
def isfinite(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isfinite(x, out=out)


isfinite.support_native_out = True


@_handle_0_dim_output
def isinf(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isinf(x, out=out)


isinf.support_native_out = True


@_handle_0_dim_output
def isnan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isnan(x, out=out)


isnan.support_native_out = True


@_handle_0_dim_output
def less(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less(x1, x2, out=out)


less.support_native_out = True


@_handle_0_dim_output
def less_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@_handle_0_dim_output
def log(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log(x, out=out)


log.support_native_out = True


@_handle_0_dim_output
def log10(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log10(x, out=out)


log10.support_native_out = True


@_handle_0_dim_output
def log1p(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log1p(x, out=out)


log1p.support_native_out = True


@_handle_0_dim_output
def log2(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log2(x, out=out)


log2.support_native_out = True


@_handle_0_dim_output
def logaddexp(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


@_handle_0_dim_output
def logical_and(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_and(x1, x2, out=out)


logical_and.support_native_out = True


@_handle_0_dim_output
def logical_not(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.logical_not(x, out=out)


logical_not.support_native_out = True


@_handle_0_dim_output
def logical_or(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_or(x1, x2, out=out)


logical_or.support_native_out = True


@_handle_0_dim_output
def logical_xor(
    x1: np.ndarray, x2: np.ndarray, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_xor(x1, x2, out=out)


logical_xor.support_native_out = True


@_handle_0_dim_output
def multiply(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.multiply(x1, x2, out=out)


multiply.support_native_out = True


@_handle_0_dim_output
def negative(
    x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.negative(x, out=out)


negative.support_native_out = True


@_handle_0_dim_output
def not_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@_handle_0_dim_output
def positive(
    x: Union[float, np.ndarray], *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.positive(x, out=out)


positive.support_native_out = True


@_handle_0_dim_output
def pow(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.power(x1, x2, out=out)


pow.support_native_out = True


@_handle_0_dim_output
def remainder(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.remainder(x1, x2, out=out)


remainder.support_native_out = True


@_handle_0_dim_output
def round(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.round(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


round.support_native_out = True


@_handle_0_dim_output
def sign(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sign(x, out=out)


sign.support_native_out = True


@_handle_0_dim_output
def sin(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sin(x, out=out)


sin.support_native_out = True


@_handle_0_dim_output
def sinh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinh(x, out=out)


sinh.support_native_out = True


@_handle_0_dim_output
def sqrt(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sqrt(x, out=out)


sqrt.support_native_out = True


@_handle_0_dim_output
def square(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.square(x, out=out)


square.support_native_out = True


@_handle_0_dim_output
def subtract(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.subtract(x1, x2, out=out)


subtract.support_native_out = True


@_handle_0_dim_output
def tan(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tan(x, out=out)


tan.support_native_out = True


@_handle_0_dim_output
def tanh(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tanh(x, out=out)


tanh.support_native_out = True


@_handle_0_dim_output
def trunc(x: np.ndarray, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.trunc(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


# Extra #
# ------#


@_handle_0_dim_output
def erf(x, *, out: Optional[np.ndarray] = None):
    if _erf is None:
        raise Exception(
            "scipy must be installed in order to call ivy.erf with a numpy backend."
        )
    ret = _erf(x, out=out)
    if hasattr(x, "dtype"):
        ret = np.asarray(_erf(x, out=out), dtype=x.dtype)
    return ret


erf.support_native_out = True


@_handle_0_dim_output
def maximum(x1, x2, *, out: Optional[np.ndarray] = None):
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.maximum(x1, x2, out=out)


maximum.support_native_out = True


@_handle_0_dim_output
def minimum(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    *,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.minimum(x1, x2, out=out)


minimum.support_native_out = True
