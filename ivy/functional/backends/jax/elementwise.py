# global
from typing import Union, Optional

import jax
import jax.numpy as jnp

# local
import ivy
from ivy import (
    default_float_dtype,
    is_float_dtype,
)
from ivy import promote_types_of_inputs
from ivy.functional.backends.jax import JaxArray
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def abs(
    x: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if (hasattr(x, "dtype") and "bool" in str(x.dtype)) or isinstance(x, bool):
        return x
    # jnp.where is used for consistent gradients
    return jnp.where(x != 0, jnp.absolute(x), 0)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def acos(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccos(x)


def acosh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arccosh(x)


def add(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    alpha: Union[int, float] = 1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        with ivy.ArrayMode(False):
            x2 = multiply(x2, alpha)
    return jnp.add(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def asin(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arcsin(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def asinh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arcsinh(x)


def atan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctan(x)


def atan2(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.arctan2(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def atanh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.arctanh(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_and(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_and(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_invert(
    x: Union[int, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.bitwise_not(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_left_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.left_shift(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_or(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_or(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_right_shift(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.right_shift(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def bitwise_xor(
    x1: Union[int, JaxArray],
    x2: Union[int, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return jnp.bitwise_xor(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def ceil(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.ceil(x)


def cos(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cos(x)


@with_unsupported_dtypes({"0.4.24 and below": ("float16",)}, backend_version)
def cosh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.cosh(x)


def divide(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = jax.numpy.divide(x1, x2)
    if ivy.is_float_dtype(x1.dtype) or ivy.is_complex_dtype(x1.dtype):
        ret = jnp.asarray(ret, dtype=x1.dtype)
    else:
        ret = jnp.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret


def equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.equal(x1, x2)


def exp(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.exp(x)


def expm1(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.expm1(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def floor(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.floor(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def floor_divide(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.floor(jnp.divide(x1, x2)).astype(x1.dtype)


def fmin(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmin(x1, x2)


def greater(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.greater(x1, x2)


def greater_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.greater_equal(x1, x2)


def isfinite(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isfinite(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def isinf(
    x: JaxArray,
    /,
    *,
    detect_positive: bool = True,
    detect_negative: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if detect_positive and detect_negative:
        return jnp.isinf(x)
    elif detect_positive:
        return jnp.isposinf(x)
    elif detect_negative:
        return jnp.isneginf(x)
    return jnp.full_like(x, False, dtype=jnp.bool_)


def isnan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isnan(x)


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.lcm(x1, x2)


def less(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.less(x1, x2)


def less_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.less_equal(x1, x2)


def log(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log(x)


def log10(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log10(x)


def log1p(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log1p(x)


def log2(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.log2(x)


def logaddexp(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logaddexp(x1, x2)


def logaddexp2(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not is_float_dtype(x1):
        x1 = x1.astype(default_float_dtype(as_native=True))
        x2 = x2.astype(default_float_dtype(as_native=True))
    return jnp.logaddexp2(x1, x2)


def logical_and(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_and(x1, x2)


def logical_not(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.logical_not(x)


def logical_or(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_or(x1, x2)


def logical_xor(
    x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.logical_xor(x1, x2)


def multiply(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.multiply(x1, x2)


def nan_to_num(
    x: JaxArray,
    /,
    *,
    copy: bool = True,
    nan: Union[float, int] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


def negative(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.negative(x)


def not_equal(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return jnp.not_equal(x1, x2)


def positive(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.positive(x)


def pow(
    x1: JaxArray,
    x2: Union[int, float, JaxArray],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if (
        ivy.any(x1 == 0)
        and ivy.is_int_dtype(x1)
        and ivy.any(x2 < 0)
        and all(dtype not in str(x1.dtype) for dtype in ["int16", "int8"])
    ):
        if ivy.is_int_dtype(x1):
            fill_value = jnp.iinfo(x1.dtype).min
        else:
            fill_value = jnp.finfo(x1.dtype).min
        ret = jnp.float_power(x1, x2)
        return jnp.where(jnp.bitwise_and(x1 == 0, x2 < 0), fill_value, ret).astype(
            x1.dtype
        )
    if ivy.is_int_dtype(x1) and ivy.any(x2 < 0):
        return jnp.float_power(x1, x2).astype(x1.dtype)
    return jnp.power(x1, x2)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def remainder(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    modulus: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = jnp.where(res >= 0, jnp.floor(res), jnp.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return jnp.round(diff * x2).astype(x1.dtype)
    return jnp.remainder(x1, x2)


def round(
    x: JaxArray, /, *, decimals: int = 0, out: Optional[JaxArray] = None
) -> JaxArray:
    if "int" in str(x.dtype):
        ret = jnp.copy(x)
    else:
        ret = jnp.round(x, decimals=decimals)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def _abs_variant_sign(x):
    return jnp.where(x != 0, x / jnp.abs(x), 0)


def sign(
    x: JaxArray, /, *, np_variant: Optional[bool] = True, out: Optional[JaxArray] = None
) -> JaxArray:
    if "complex" in str(x.dtype):
        return jnp.sign(x) if np_variant else _abs_variant_sign(x)
    return jnp.where(x == -0.0, 0.0, jnp.sign(x)).astype(x.dtype)


def sin(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sin(x)


def sinh(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinh(x)


def sqrt(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sqrt(x)


def square(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.square(x)


def subtract(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    alpha: Optional[Union[int, float]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        ivy.set_array_mode(False)
        x2 = multiply(x2, alpha)
        ivy.unset_array_mode()
    return jnp.subtract(x1, x2)


def trapz(
    y: JaxArray,
    /,
    *,
    x: Optional[JaxArray] = None,
    dx: float = 1.0,
    axis: int = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trapz(y, x=x, dx=dx, axis=axis)


@with_unsupported_dtypes(
    {"0.4.24 and below": ("complex", "float16", "bfloat16")}, backend_version
)
def tan(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.tan(x)


def tanh(
    x: JaxArray, /, *, complex_mode="jax", out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.tanh(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def trunc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    if "int" in str(x.dtype):
        return x
    else:
        return jnp.trunc(x)


def exp2(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.power(2, x)


def imag(
    val: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.imag(val)


def angle(
    z: JaxArray,
    /,
    *,
    deg: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.angle(z, deg=deg)


# Extra #
# ------#


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def erf(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jax.scipy.special.erf(x)


def maximum(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    use_where: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return jnp.where(x1 >= x2, x1, x2)
    return jnp.maximum(x1, x2)


def minimum(
    x1: Union[float, JaxArray],
    x2: Union[float, JaxArray],
    /,
    *,
    use_where: bool = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if use_where:
        return jnp.where(x1 <= x2, x1, x2)
    return jnp.minimum(x1, x2)


def reciprocal(
    x: Union[float, JaxArray], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.reciprocal(x)


def deg2rad(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.deg2rad(x)


def rad2deg(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.rad2deg(x)


def isreal(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.isreal(x)


@with_unsupported_dtypes({"0.4.24 and below": ("complex",)}, backend_version)
def fmod(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.fmod(x1, x2)


def gcd(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return jnp.gcd(x1, x2)


def real(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.real(x)
