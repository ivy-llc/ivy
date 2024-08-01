# global
from typing import Union, Optional
import keras
import jax
import tensorflow as tf
import torch

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
)
from ivy import promote_types_of_inputs
from . import backend_version
from .func_wrapper import use_keras_backend_framework


@use_keras_backend_framework
def abs(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def acos(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def acosh(
    x,
    /,
    *,
    out=None
):
    pass


@with_unsupported_dtypes({"3.4.1 and below": ("complex",)}, backend_version)
def add(
    x1: Union[float, jax.Array, tf.Tensor, torch.Tensor],
    x2: Union[float, jax.Array, tf.Tensor, torch.Tensor],
    /,
    *,
    alpha=None,
    out: Optional[Union[jax.Array, tf.Tensor, torch.Tensor]] = None,
) -> Union[jax.Array, tf.Tensor, torch.Tensor]:
    x1, x2 = promote_types_of_inputs(x1, x2)

    if alpha is not None:
        x2 = keras.ops.multiply(x2, alpha)

    ret = keras.ops.add(x1, x2)

    if ivy.exists(out):
        ivy.inplace_update(out, ret)
    return ret


add.support_native_out = True


@use_keras_backend_framework
def asin(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def asinh(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def atan(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def atan2(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def atanh(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_and(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_invert(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_left_shift(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_or(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_right_shift(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def bitwise_xor(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def ceil(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def cos(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def cosh(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def divide(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def equal(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
@with_unsupported_dtypes({"2.15.0 and below": ("integer",)}, backend_version)
def exp(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def exp2(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def expm1(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def floor(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def floor_divide(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def fmin(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def greater(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def greater_equal(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def isfinite(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def isinf(
    x,
    /,
    *,
    detect_positive=True,
    detect_negative=True,
    out=None
):
    pass


@use_keras_backend_framework
def isnan(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def lcm(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def less(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def less_equal(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def log(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def log10(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def log1p(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def log2(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logaddexp(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def real(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logaddexp2(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logical_and(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logical_not(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logical_or(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def logical_xor(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def multiply(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def negative(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def not_equal(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def positive(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def pow(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def remainder(
    x1,
    x2,
    /,
    *,
    modulus=True,
    out=None
):
    pass


@use_keras_backend_framework
def round(
    x,
    /,
    *,
    decimals=0,
    out=None
):
    pass


@use_keras_backend_framework
def sign(
    x,
    /,
    *,
    np_variant=True,
    out=None
):
    pass


@use_keras_backend_framework
def sin(
    x,
    /,
    *,
    out=None
):
    return tf.sin(x)


@use_keras_backend_framework
def sinh(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def sqrt(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def square(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def subtract(
    x1,
    x2,
    /,
    *,
    alpha=None,
    out=None
):
    pass


@use_keras_backend_framework
def tan(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def tanh(
    x,
    /,
    *,
    complex_mode="jax",
    out=None
):
    pass


@use_keras_backend_framework
def trapz(
    y,
    /,
    *,
    x=None,
    dx=1.0,
    axis=-1,
    out=None
):
    pass


@use_keras_backend_framework
def trunc(
    x,
    /,
    *,
    out=None
):
    pass


# Extra #
# ------#


@use_keras_backend_framework
def erf(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def maximum(
    x1,
    x2,
    /,
    *,
    use_where=True,
    out=None
):
    pass


@use_keras_backend_framework
def minimum(
    x1,
    x2,
    /,
    *,
    use_where=True,
    out=None
):
    pass


@use_keras_backend_framework
def reciprocal(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def deg2rad(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def rad2deg(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def isreal(
    x,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def fmod(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def gcd(
    x1,
    x2,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def angle(
    input,
    /,
    *,
    deg=None,
    out=None
):
    pass


@use_keras_backend_framework
def imag(
    val,
    /,
    *,
    out=None
):
    pass


@use_keras_backend_framework
def nan_to_num(
    x,
    /,
    *,
    copy=True,
    nan=0.0,
    posinf=None,
    neginf=None,
    out=None
):
    pass
