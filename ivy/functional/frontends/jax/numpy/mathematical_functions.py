# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_ivy_arrays_and_back
def absolute(x):
    return ivy.abs(x)


abs = absolute


@to_ivy_arrays_and_back
def add(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.add(x1, x2)


@to_ivy_arrays_and_back
def arctan(x):
    ret = ivy.atan(x)
    return ret


@to_ivy_arrays_and_back
def arctan2(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.atan2(x1, x2)


@to_ivy_arrays_and_back
def cos(x):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def cosh(x):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def dot(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def floor(x):
    return ivy.floor(x)


@to_ivy_arrays_and_back
def mod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def sinh(x):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def sin(x):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def tan(x):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def tanh(x):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def arccos(x):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def arccosh(x):
    return ivy.acosh(x)


@to_ivy_arrays_and_back
def arcsin(x):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def arcsinh(x):
    return ivy.asinh(x)


@to_ivy_arrays_and_back
def power(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.pow(x1, x2)


@to_ivy_arrays_and_back
def trunc(x):
    return ivy.trunc(x)


@to_ivy_arrays_and_back
def ceil(x):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def float_power(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.float_power(x1, x2)


@to_ivy_arrays_and_back
def deg2rad(x):
    return ivy.deg2rad(x)


@to_ivy_arrays_and_back
def radians(x):
    return ivy.deg2rad(x)


@to_ivy_arrays_and_back
def exp2(x):
    return ivy.exp2(x)


@to_ivy_arrays_and_back
def gcd(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.gcd(x1, x2)


@to_ivy_arrays_and_back
def i0(x):
    return ivy.i0(x)


@to_ivy_arrays_and_back
def kron(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.kron(a, b)


@to_ivy_arrays_and_back
def lcm(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.lcm(x1, x2)


@to_ivy_arrays_and_back
def logaddexp2(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.logaddexp2(x1, x2)


@to_ivy_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1, out=None):
    return ivy.trapz(y, x=x, dx=dx, axis=axis, out=out)


@to_ivy_arrays_and_back
def sqrt(x, /):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def square(x, /):
    return ivy.square(x)


@to_ivy_arrays_and_back
def arctanh(x):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def multiply(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.multiply(x1, x2)


@to_ivy_arrays_and_back
def matmul(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def log10(x):
    return ivy.log10(x)


@to_ivy_arrays_and_back
def logaddexp(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.logaddexp(x1, x2)


@to_ivy_arrays_and_back
def degrees(x):
    return ivy.rad2deg(x)


@to_ivy_arrays_and_back
def negative(
    x,
    /,
):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def rad2deg(
    x,
    /,
):
    return ivy.rad2deg(x)


@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.tensordot(a, b, axes=axes)


@to_ivy_arrays_and_back
def divide(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    if ivy.dtype(x1) in ["int64", "uint64"]:
        x1 = ivy.astype(x1, ivy.float64)
    elif ivy.is_int_dtype(x1):
        x1 = ivy.astype(x1, ivy.float32)

    return ivy.divide(x1, x2).astype(x1.dtype)


true_divide = divide


@to_ivy_arrays_and_back
def exp(
    x,
    /,
):
    return ivy.exp(x)


def expm1(
    x,
    /,
):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def fmax(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    ret = ivy.where(
        ivy.bitwise_or(ivy.greater(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    return ret


@to_ivy_arrays_and_back
def fmin(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    ret = ivy.where(
        ivy.bitwise_or(ivy.less(x1, x2), ivy.isnan(x2)),
        x1,
        x2,
    )
    print("jax-frontend", ret)
    return ret


@with_unsupported_dtypes(
    {"0.3.14 and below": ("uint16",)},
    "jax",
)
@to_ivy_arrays_and_back
def fabs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def fmod(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.fmod(x1, x2)


@to_ivy_arrays_and_back
def maximum(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.maximum(x1, x2)


@to_ivy_arrays_and_back
def minimum(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.minimum(x1, x2)


@to_ivy_arrays_and_back
def heaviside(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return ivy.heaviside(x1, x2)


@to_ivy_arrays_and_back
def log(x):
    return ivy.log(x)


@to_ivy_arrays_and_back
def log1p(x, /):
    return ivy.log1p(x)


@to_ivy_arrays_and_back
def copysign(x1, x2):
    return ivy.copysign(x1, x2)


@to_ivy_arrays_and_back
def sinc(x):
    return ivy.sinc(x)


@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "bfloat16",
            "float16",
        )
    },
    "jax",
)
@to_ivy_arrays_and_back
def nextafter(x1, x2):
    return ivy.nextafter(x1, x2)


@to_ivy_arrays_and_back
def remainder(x1, x2):
    return ivy.remainder(x1, x2)


@to_ivy_arrays_and_back
def trace(a, offset=0, axis1=0, axis2=1, out=None):
    return ivy.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)


@to_ivy_arrays_and_back
def log2(x):
    return ivy.log2(x)


@to_ivy_arrays_and_back
def vdot(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return ivy.multiply(a, b).sum()


@with_unsupported_dtypes(
    {"0.3.14 and below": ("bfloat16",)},
    "jax",
)
@to_ivy_arrays_and_back
def cbrt(x, /):
    all_positive = ivy.pow(ivy.abs(x), 1.0 / 3.0)
    return ivy.where(ivy.less(x, 0.0), ivy.negative(all_positive), all_positive)


@to_ivy_arrays_and_back
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return ivy.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


@to_ivy_arrays_and_back
def fix(x, out=None):
    return ivy.fix(x, out=out)


@to_ivy_arrays_and_back
def real(val, /):
    return ivy.real(val)


@to_ivy_arrays_and_back
def hypot(x1, x2, /):
    return ivy.hypot(x1, x2)


@to_ivy_arrays_and_back
def floor_divide(x1, x2, /, out=None):
    return ivy.floor_divide(x1, x2, out=out)
