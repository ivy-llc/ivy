# global
from typing import Union, Optional

import cupy as cp

# local
import ivy
from ivy.functional.backends.numpy.helpers import _handle_0_dim_output

try:
    from scipy.special import erf as _erf
except (ImportError, ModuleNotFoundError):
    _erf = None


# when inputs are 0 dimensional, cupy's functions return scalars
# so we use this wrapper to ensure outputs are always numpy/cupy arrays


@_handle_0_dim_output
def abs(
    x: Union[float, cp.ndarray], /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.absolute(x, out=out)


abs.support_native_out = True


@_handle_0_dim_output
def acos(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arccos(x, out=out)


acos.support_native_out = True


@_handle_0_dim_output
def acosh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arccosh(x, out=out)


acosh.support_native_out = True


@_handle_0_dim_output
def add(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return cp.add(x1, x2)


add.support_native_out = True


@_handle_0_dim_output
def asin(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arcsin(x, out=out)


asin.support_native_out = True


@_handle_0_dim_output
def asinh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arcsinh(x, out=out)


asinh.support_native_out = True


@_handle_0_dim_output
def atan(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arctan(x, out=out)


atan.support_native_out = True


@_handle_0_dim_output
def atan2(
    x1: cp.ndarray, x2: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.arctan2(x1, x2, out=out)


atan2.support_native_out = True


@_handle_0_dim_output
def atanh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.arctanh(x, out=out)


atanh.support_native_out = True


@_handle_0_dim_output
def bitwise_and(
    x1: Union[int, bool, cp.ndarray],
    x2: Union[int, bool, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return cp.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@_handle_0_dim_output
def bitwise_invert(
    x: Union[int, bool, cp.ndarray], /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.invert(x, out=out)


bitwise_invert.support_native_out = True


@_handle_0_dim_output
def bitwise_left_shift(
    x1: Union[int, bool, cp.ndarray],
    x2: Union[int, bool, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    ivy.assertions.check_all(x2 >= 0, message="shifts must be non-negative")
    return cp.left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


@_handle_0_dim_output
def bitwise_or(
    x1: Union[int, bool, cp.ndarray],
    x2: Union[int, bool, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return cp.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@_handle_0_dim_output
def bitwise_right_shift(
    x1: Union[int, bool, cp.ndarray],
    x2: Union[int, bool, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    ivy.assertions.check_all(x2 >= 0, message="shifts must be non-negative")
    return cp.right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@_handle_0_dim_output
def bitwise_xor(
    x1: Union[int, bool, cp.ndarray],
    x2: Union[int, bool, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return cp.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


@_handle_0_dim_output
def ceil(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


ceil.support_native_out = True


@_handle_0_dim_output
def cos(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.cos(x, out=out)


cos.support_native_out = True


@_handle_0_dim_output
def cosh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.cosh(x, out=out)


cosh.support_native_out = True


@_handle_0_dim_output
def divide(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = cp.divide(x1, x2)
    if ivy.is_float_dtype(x1):
        ret = cp.asarray(ret, dtype=x1.dtype)
    else:
        ret = cp.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


divide.support_native_out = True


@_handle_0_dim_output
def equal(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.equal(x1, x2, out=out)


equal.support_native_out = True


@_handle_0_dim_output
def exp(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.exp(x, out=out)


exp.support_native_out = True


@_handle_0_dim_output
def expm1(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.expm1(x, out=out)


expm1.support_native_out = True


@_handle_0_dim_output
def floor(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


floor.support_native_out = True


@_handle_0_dim_output
def floor_divide(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.floor(cp.divide(x1, x2)).astype(x1.dtype)


@_handle_0_dim_output
def greater(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.greater(x1, x2, out=out)


greater.support_native_out = True


@_handle_0_dim_output
def greater_equal(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@_handle_0_dim_output
def isfinite(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isfinite(x, out=out)


isfinite.support_native_out = True


@_handle_0_dim_output
def isinf(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isinf(x, out=out)


isinf.support_native_out = True


@_handle_0_dim_output
def isnan(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.isnan(x, out=out)


isnan.support_native_out = True


@_handle_0_dim_output
def less(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.less(x1, x2, out=out)


less.support_native_out = True


@_handle_0_dim_output
def less_equal(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@_handle_0_dim_output
def log(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log(x, out=out)


log.support_native_out = True


@_handle_0_dim_output
def log10(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log10(x, out=out)


log10.support_native_out = True


@_handle_0_dim_output
def log1p(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log1p(x, out=out)


log1p.support_native_out = True


@_handle_0_dim_output
def log2(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.log2(x, out=out)


log2.support_native_out = True


@_handle_0_dim_output
def logaddexp(
    x1: cp.ndarray, x2: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


@_handle_0_dim_output
def logical_and(
    x1: cp.ndarray, x2: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_and(x1, x2, out=out)


logical_and.support_native_out = True


@_handle_0_dim_output
def logical_not(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.logical_not(x, out=out)


logical_not.support_native_out = True


@_handle_0_dim_output
def logical_or(
    x1: cp.ndarray, x2: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_or(x1, x2, out=out)


logical_or.support_native_out = True


@_handle_0_dim_output
def logical_xor(
    x1: cp.ndarray, x2: cp.ndarray, /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.logical_xor(x1, x2, out=out)


logical_xor.support_native_out = True


@_handle_0_dim_output
def multiply(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.multiply(x1, x2, out=out)


multiply.support_native_out = True


@_handle_0_dim_output
def negative(
    x: Union[float, cp.ndarray], /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.negative(x, out=out)


negative.support_native_out = True


@_handle_0_dim_output
def not_equal(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    return cp.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@_handle_0_dim_output
def positive(
    x: Union[float, cp.ndarray], /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.positive(x, out=out)


positive.support_native_out = True


@_handle_0_dim_output
def pow(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.power(x1, x2, out=out)


pow.support_native_out = True


@_handle_0_dim_output
def remainder(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    modulus: bool = True,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = cp.where(res >= 0, cp.floor(res), cp.ceil(res))
        diff = cp.asarray(res - res_floored, dtype=res.dtype)
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return cp.asarray(cp.round(diff * x2), dtype=x1.dtype)
    return cp.remainder(x1, x2, out=out)


remainder.support_native_out = True


@_handle_0_dim_output
def round(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.round(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


round.support_native_out = True


@_handle_0_dim_output
def sign(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sign(x, out=out)


sign.support_native_out = True


@_handle_0_dim_output
def sin(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sin(x, out=out)


sin.support_native_out = True


@_handle_0_dim_output
def sinh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sinh(x, out=out)


sinh.support_native_out = True


@_handle_0_dim_output
def sqrt(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.sqrt(x, out=out)


sqrt.support_native_out = True


@_handle_0_dim_output
def square(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.square(x, out=out)


square.support_native_out = True


@_handle_0_dim_output
def subtract(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = multiply(x2, alpha)
    return cp.subtract(x1, x2)


subtract.support_native_out = True


@_handle_0_dim_output
def tan(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.tan(x, out=out)


tan.support_native_out = True


@_handle_0_dim_output
def tanh(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.tanh(x, out=out)


tanh.support_native_out = True


@_handle_0_dim_output
def trunc(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    if "int" in str(x.dtype):
        ret = cp.copy(x)
    else:
        return cp.trunc(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


# Extra #
# ------#


@_handle_0_dim_output
def erf(x, /, *, out: Optional[cp.ndarray] = None):
    ivy.assertions.check_exists(
        _erf,
        message="scipy must be installed in order to call ivy.erf with a \
        numpy backend.",
    )
    ret = _erf(x, out=out)
    if hasattr(x, "dtype"):
        ret = cp.asarray(_erf(x, out=out), dtype=x.dtype)
    return ret


erf.support_native_out = True


@_handle_0_dim_output
def maximum(x1, x2, /, *, out: Optional[cp.ndarray] = None):
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.maximum(x1, x2, out=out)


maximum.support_native_out = True


@_handle_0_dim_output
def minimum(
    x1: Union[float, cp.ndarray],
    x2: Union[float, cp.ndarray],
    /,
    *,
    out: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return cp.minimum(x1, x2, out=out)


minimum.support_native_out = True


@_handle_0_dim_output
def reciprocal(
    x: Union[float, cp.ndarray], /, *, out: Optional[cp.ndarray] = None
) -> cp.ndarray:
    return cp.reciprocal(x, out=out)


reciprocal.support_native_out = True


@_handle_0_dim_output
def deg2rad(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.deg2rad(x, out=out)


deg2rad.support_native_out = True


@_handle_0_dim_output
def rad2deg(x: cp.ndarray, /, *, out: Optional[cp.ndarray] = None) -> cp.ndarray:
    return cp.rad2deg(x, out=out)


rad2deg.support_native_out = True
