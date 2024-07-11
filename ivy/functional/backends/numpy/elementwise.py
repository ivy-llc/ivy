# global
from typing import Union, Optional
import numpy as np

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy import promote_types_of_inputs
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from . import backend_version


@_scalar_output_to_0d_array
def abs(
    x: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.absolute(x, out=out)


abs.support_native_out = True


@_scalar_output_to_0d_array
def acos(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccos(x, out=out)


acos.support_native_out = True


@_scalar_output_to_0d_array
def acosh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arccosh(x, out=out)


acosh.support_native_out = True


@_scalar_output_to_0d_array
def add(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        with ivy.ArrayMode(False):
            x2 = multiply(x2, alpha)
    return np.add(x1, x2, out=out)


add.support_native_out = True


@_scalar_output_to_0d_array
def asin(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsin(x, out=out)


asin.support_native_out = True


@_scalar_output_to_0d_array
def asinh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arcsinh(x, out=out)


asinh.support_native_out = True


@_scalar_output_to_0d_array
def atan(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctan(x, out=out)


atan.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def atan2(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.arctan2(x1, x2, out=out)


atan2.support_native_out = True


@_scalar_output_to_0d_array
def atanh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.arctanh(x, out=out)


atanh.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_and(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return np.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_invert(
    x: Union[int, bool, np.ndarray], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.invert(x, out=out)


bitwise_invert.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_left_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return np.left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_or(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return np.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_right_shift(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return np.right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def bitwise_xor(
    x1: Union[int, bool, np.ndarray],
    x2: Union[int, bool, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return np.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
@_scalar_output_to_0d_array
def ceil(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.ceil(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


ceil.support_native_out = True


@_scalar_output_to_0d_array
def cos(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cos(x, out=out)


cos.support_native_out = True


@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, backend_version)
@_scalar_output_to_0d_array
def cosh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.cosh(x, out=out)


cosh.support_native_out = True


@_scalar_output_to_0d_array
def divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = np.divide(x1, x2, out=out)
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
        ret = np.asarray(ret, dtype=x1.dtype)
    else:
        ret = np.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


divide.support_native_out = True


@_scalar_output_to_0d_array
def equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.equal(x1, x2, out=out)


equal.support_native_out = True


@_scalar_output_to_0d_array
def exp(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.exp(x, out=out)


exp.support_native_out = True


def exp2(
    x: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.exp2(x, out=out)


exp2.support_native_out = True


@_scalar_output_to_0d_array
def expm1(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.expm1(x, out=out)


expm1.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def floor(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        return np.floor(x, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


floor.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def floor_divide(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.floor(np.divide(x1, x2)).astype(x1.dtype)


@_scalar_output_to_0d_array
def fmin(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.fmin(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    )


fmin.support_native_out = True


@_scalar_output_to_0d_array
def greater(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.greater(x1, x2, out=out)


greater.support_native_out = True


@_scalar_output_to_0d_array
def greater_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


@_scalar_output_to_0d_array
def isfinite(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isfinite(x, out=out)


isfinite.support_native_out = True


@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
@_scalar_output_to_0d_array
def isinf(
    x: np.ndarray,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if detect_negative and detect_positive:
        return np.isinf(x)
    elif detect_negative:
        return np.isneginf(x)
    elif detect_positive:
        return np.isposinf(x)
    return np.full_like(x, False, dtype=bool)


@_scalar_output_to_0d_array
def isnan(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isnan(x, out=out)


isnan.support_native_out = True


@_scalar_output_to_0d_array
def lcm(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.lcm(
        x1,
        x2,
        out=out,
    )


lcm.support_native_out = True


@_scalar_output_to_0d_array
def less(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.less(x1, x2, out=out)


less.support_native_out = True


@_scalar_output_to_0d_array
def less_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


@_scalar_output_to_0d_array
def log(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log(x, out=out)


log.support_native_out = True


@_scalar_output_to_0d_array
def log10(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log10(x, out=out)


log10.support_native_out = True


@_scalar_output_to_0d_array
def log1p(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log1p(x, out=out)


log1p.support_native_out = True


@_scalar_output_to_0d_array
def log2(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.log2(x, out=out)


log2.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def logaddexp(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True


def logaddexp2(
    x1: Union[np.ndarray, int, list, tuple],
    x2: Union[np.ndarray, int, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.astype(ivy.default_float_dtype(as_native=True))
        x2 = x2.astype(ivy.default_float_dtype(as_native=True))
    return np.logaddexp2(x1, x2, out=out)


logaddexp2.support_native_out = True


@_scalar_output_to_0d_array
def logical_and(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_and(x1, x2, out=out)


logical_and.support_native_out = True


@_scalar_output_to_0d_array
def logical_not(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.logical_not(x, out=out)


logical_not.support_native_out = True


@_scalar_output_to_0d_array
def logical_or(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_or(x1, x2, out=out)


logical_or.support_native_out = True


@_scalar_output_to_0d_array
def logical_xor(
    x1: np.ndarray, x2: np.ndarray, /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.logical_xor(x1, x2, out=out)


logical_xor.support_native_out = True


@_scalar_output_to_0d_array
def multiply(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.multiply(x1, x2, out=out)


multiply.support_native_out = True


@_scalar_output_to_0d_array
def negative(
    x: Union[float, np.ndarray], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.negative(x, out=out)


negative.support_native_out = True


@_scalar_output_to_0d_array
def not_equal(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


@_scalar_output_to_0d_array
def positive(
    x: Union[float, np.ndarray], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.positive(x, out=out)


positive.support_native_out = True


@_scalar_output_to_0d_array
def pow(
    x1: np.ndarray,
    x2: Union[int, float, np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if ivy.is_complex_dtype(x1) and ivy.any(ivy.isinf(x2)):
        ret = np.power(x1, x2)
        return np.where(np.isinf(x2), np.nan + np.nan * 1j if x2 < 0 else -0 * 1j, ret)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if ivy.is_int_dtype(x1) and ivy.any(x2 < 0):
        return np.float_power(x1, x2, casting="unsafe").astype(x1.dtype)
    return np.power(x1, x2)


pow.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def remainder(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    modulus: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = np.where(res >= 0, np.floor(res), np.ceil(res))
        diff = np.asarray(res - res_floored, dtype=res.dtype)
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return np.asarray(np.round(diff * x2), dtype=x1.dtype)
    return np.remainder(x1, x2, out=out)


remainder.support_native_out = True


@_scalar_output_to_0d_array
def round(
    x: np.ndarray, /, *, decimals: int = 0, out: Optional[np.ndarray] = None
) -> np.ndarray:
    if "int" in str(x.dtype):
        ret = np.copy(x)
    else:
        ret = np.round(x, decimals=decimals, out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


round.support_native_out = True


def _abs_variant_sign(x):
    return np.divide(x, np.abs(x), where=x != 0)


@_scalar_output_to_0d_array
def sign(
    x: np.ndarray,
    /,
    *,
    np_variant: Optional[bool] = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if "complex" in str(x.dtype):
        return np.sign(x, out=out) if np_variant else _abs_variant_sign(x)
    return np.sign(x, out=out)


sign.support_native_out = True


@_scalar_output_to_0d_array
def sin(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sin(x, out=out)


sin.support_native_out = True


@_scalar_output_to_0d_array
def sinh(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinh(x, out=out)


sinh.support_native_out = True


@_scalar_output_to_0d_array
def sqrt(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sqrt(x, out=out)


sqrt.support_native_out = True


@_scalar_output_to_0d_array
def square(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.square(x, out=out)


square.support_native_out = True


@_scalar_output_to_0d_array
def subtract(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        ivy.set_array_mode(False)
        x2 = multiply(x2, alpha)
        ivy.unset_array_mode()
    return np.subtract(x1, x2, out=out)


subtract.support_native_out = True


@_scalar_output_to_0d_array
def trapz(
    y: np.ndarray,
    /,
    *,
    x: Optional[np.ndarray] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.trapz(y, x=x, dx=dx, axis=axis)


trapz.support_native_out = False


@_scalar_output_to_0d_array
def tan(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.tan(x, out=out)


tan.support_native_out = True


@_scalar_output_to_0d_array
def tanh(
    x: np.ndarray, /, *, complex_mode="jax", out: Optional[np.ndarray] = None
) -> np.ndarray:
    return np.tanh(x, out=out)


tanh.support_native_out = True


@_scalar_output_to_0d_array
def trunc(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
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


@_scalar_output_to_0d_array
def erf(x, /, *, out: Optional[np.ndarray] = None):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    ret = sign * y
    if hasattr(x, "dtype"):
        ret = np.asarray(ret, dtype=x.dtype)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


erf.support_native_out = True


@_scalar_output_to_0d_array
def maximum(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    use_where: bool = True,
    out: Optional[np.ndarray] = None,
):
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        ret = np.where(x1 >= x2, x1, x2)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    return np.maximum(x1, x2, out=out)


maximum.support_native_out = True


@_scalar_output_to_0d_array
def minimum(
    x1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    /,
    *,
    use_where: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        ret = np.where(x1 <= x2, x1, x2)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    return np.minimum(x1, x2, out=out)


minimum.support_native_out = True


@_scalar_output_to_0d_array
def reciprocal(
    x: Union[float, np.ndarray], /, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    numerator = np.ones_like(x)
    return np.true_divide(numerator, x, out=out)


reciprocal.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def deg2rad(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.deg2rad(x, out=out)


deg2rad.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def rad2deg(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.rad2deg(x, out=out)


rad2deg.support_native_out = True


@_scalar_output_to_0d_array
def isreal(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.isreal(x)


isreal.support_native_out = False


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def fmod(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.fmod(
        x1,
        x2,
        out=None,
    )


fmod.support_native_out = True


def angle(
    z: np.ndarray,
    /,
    *,
    deg: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.angle(z, deg=deg)


angle.support_native_out = False


def gcd(
    x1: Union[np.ndarray, int, list, tuple],
    x2: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return np.gcd(x1, x2, out=out)


gcd.support_native_out = True


def imag(
    val: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.imag(val)


imag.support_native_out = False


def nan_to_num(
    x: np.ndarray,
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


nan_to_num.support_native_out = False


def real(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.real(x)
