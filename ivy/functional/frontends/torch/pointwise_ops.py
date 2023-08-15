# global
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
)
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_gradients,
)


@with_supported_dtypes(
    {"1.12.0 and below": ("float32", "float64", "int32", "int64")}, "jax"
)
@handle_gradients
@to_ivy_arrays_and_back
def add(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.add(input, other, alpha=alpha, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def tan(input, *, out=None):
    return ivy.tan(input, out=out)


@to_ivy_arrays_and_back
def remainder(input, other, *, out=None):
    if ivy.is_array(input) and ivy.isscalar(other):
        other = ivy.full(input.shape, other)
    return ivy.remainder(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def atan(input, *, out=None):
    return ivy.atan(input, out=out)


arctan = atan


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@handle_gradients
@to_ivy_arrays_and_back
def cos(input, *, out=None):
    return ivy.cos(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@handle_gradients
@to_ivy_arrays_and_back
def sin(input, *, out=None):
    return ivy.sin(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def acos(input, *, out=None):
    return ivy.acos(input, out=out)


arccos = acos


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def acosh(input, *, out=None):
    return ivy.acosh(input, out=out)


arccosh = acosh


@to_ivy_arrays_and_back
def abs(input, *, out=None):
    return ivy.abs(input, out=out)


absolute = abs


@to_ivy_arrays_and_back
def cosh(input, *, out=None):
    return ivy.cosh(input, out=out)


@to_ivy_arrays_and_back
def subtract(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.subtract(input, other * alpha, out=out)


sub = subtract


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@handle_gradients
@to_ivy_arrays_and_back
def exp(input, *, out=None):
    return ivy.exp(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def asin(input, *, out=None):
    return ivy.asin(input, out=out)


arcsin = asin


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


arctanh = atanh


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log2(input, *, out=None):
    return ivy.log2(input, out=out)


@handle_gradients
@to_ivy_arrays_and_back
def square(input, *, out=None):
    return ivy.square(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def atan2(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.atan2(input, other, out=out)


arctan2 = atan2


@with_unsupported_dtypes({"2.0.1 and below": ("bool",)}, "torch")
@to_ivy_arrays_and_back
def negative(input, *, out=None):
    return ivy.negative(input, out=out)


@to_ivy_arrays_and_back
def bitwise_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_and(input, other, out=out)


@to_ivy_arrays_and_back
def bitwise_not(input, *, out=None):
    return ivy.bitwise_invert(input, out=out)


@to_ivy_arrays_and_back
def bitwise_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_xor(input, other, out=out)


@to_ivy_arrays_and_back
def bitwise_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_or(input, other, out=out)


@to_ivy_arrays_and_back
def bitwise_left_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_left_shift(input, other, out=out)


@to_ivy_arrays_and_back
def bitwise_right_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_right_shift(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log10(input, *, out=None):
    return ivy.log10(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def trunc(input, *, out=None):
    return ivy.trunc(input, out=out)


fix = trunc


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sqrt(input, *, out=None):
    return ivy.sqrt(input, out=out)


@to_ivy_arrays_and_back
def real(input):
    return ivy.real(input)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
@to_ivy_arrays_and_back
def sign(input, *, out=None):
    return ivy.sign(input, out=out)


@to_ivy_arrays_and_back
def logical_not(input, *, out=None):
    return ivy.logical_not(input, out=out)


@to_ivy_arrays_and_back
def logical_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_and(input, other, out=out)


@to_ivy_arrays_and_back
def logical_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_or(input, other, out=out)


@to_ivy_arrays_and_back
def logical_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_xor(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
@to_ivy_arrays_and_back
def round(input, *, decimals=0, out=None):
    m = ivy.full(input.shape, 10.0**decimals)
    upscale = ivy.multiply(input, m)
    rounded = ivy.round(upscale)
    return ivy.divide(rounded, m, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def ceil(input, *, out=None):
    return ivy.ceil(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
@to_ivy_arrays_and_back
def clamp(input, min=None, max=None, *, out=None):
    ivy.utils.assertions.check_all_or_any_fn(
        min,
        max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    if min is None:
        return ivy.minimum(input, max, out=out)
    if max is None:
        return ivy.maximum(input, min, out=out)
    return ivy.clip(input, min, max, out=out)


clip = clamp


@handle_gradients
@to_ivy_arrays_and_back
def mul(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.multiply(input, other, out=out)


multiply = mul


@to_ivy_arrays_and_back
def div(input, other, *, rounding_mode=None, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    if rounding_mode is not None:
        promoted = input.dtype
        if rounding_mode == "trunc":
            return ivy.astype(ivy.trunc_divide(input, other, out=out), promoted)
        else:
            return ivy.astype(ivy.floor_divide(input, other, out=out), promoted)
    else:
        return ivy.divide(input, other, out=out)


@to_ivy_arrays_and_back
def reciprocal(input, *, out=None):
    return ivy.reciprocal(input)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def floor(input, *, out=None):
    return ivy.floor(input, out=out)


@to_ivy_arrays_and_back
def flipud(input):
    return ivy.flipud(input)


@to_ivy_arrays_and_back
def deg2rad(input, *, out=None):
    return ivy.array(input * 3.1416 / 180, out=out)


arcsinh = asinh


divide = div


@to_ivy_arrays_and_back
def true_divide(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.divide(input, other, out=out)


@to_ivy_arrays_and_back
def floor_divide(input, other, *, out=None):
    return ivy.floor_divide(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log1p(input, *, out=None):
    return ivy.log1p(input, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def addcdiv(input, tensor1, tensor2, *, value=1, out=None):
    return ivy.add(input, ivy.multiply(value, ivy.divide(tensor1, tensor2)), out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    return ivy.add(input, ivy.multiply(value, ivy.multiply(tensor1, tensor2)), out=out)


@handle_gradients
@to_ivy_arrays_and_back
def pow(input, exponent, *, out=None):
    return ivy.pow(input, exponent, out=out)


@with_unsupported_dtypes({"1.12.0 and below": ("bfloat16", "float16")}, "jax")
@to_ivy_arrays_and_back
def float_power(input, exponent, *, out=None):
    input, exponent = torch_frontend.promote_types_of_torch_inputs(input, exponent)
    return ivy.float_power(input, exponent, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def log(input, *, out=None):
    return ivy.log(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def logaddexp(x1, x2, out=None):
    return ivy.logaddexp(x1, x2, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def exp2(input, out=None):
    return ivy.exp2(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def rsqrt(input, *, out=None):
    return ivy.reciprocal(ivy.sqrt(input), out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def expm1(input, out=None):
    return ivy.expm1(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def logaddexp2(x1, x2, out=None):
    return ivy.logaddexp2(x1, x2, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def i0(input, *, out=None):
    return ivy.i0(input, out=out)


@to_ivy_arrays_and_back
def rad2deg(input, *, out=None):
    return ivy.rad2deg(input, out=out)


@to_ivy_arrays_and_back
def positive(input, *, out=None):
    return ivy.positive(input, out=out)


@to_ivy_arrays_and_back
def frac(input, *, out=None):
    return input - ivy.sign(input) * ivy.floor(ivy.abs(input))


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def xlogy(input, other, *, out=None):
    return ivy.xlogy(input, other, out=out)


@with_unsupported_dtypes({"1.12.0 and below": ("float16",)}, "jax")
@to_ivy_arrays_and_back
def copysign(input, other, *, out=None):
    return ivy.copysign(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sinc(input, *, out=None):
    return ivy.sinc(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def hypot(input, other, *, out=None):
    return ivy.hypot(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sigmoid(input, *, out=None):
    return ivy.sigmoid(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def lerp(input, end, weight, *, out=None):
    return ivy.lerp(input, end, weight, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
@to_ivy_arrays_and_back
def signbit(input, *, out=None):
    return ivy.signbit(input, out=out)


@to_ivy_arrays_and_back
def angle(input, *, out=None):
    return ivy.angle(input, out=out)


@to_ivy_arrays_and_back
def conj_physical(input, *, out=None):
    return ivy.conj(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
@to_ivy_arrays_and_back
def nextafter(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.nextafter(input, other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def fmod(x1, x2, out=None):
    return ivy.fmod(x1, x2, out=out)


@to_ivy_arrays_and_back
def imag(input):
    return ivy.imag(input)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def logit(input, eps=None, *, out=None):
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps

    input = ivy.clip(input, lo, hi, out=out)

    return ivy.log(ivy.divide(input, ivy.subtract(1, input), out=out), out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def frexp(input, *, out=None):
    return ivy.frexp(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
@to_ivy_arrays_and_back
def erf(input, *, out=None):
    return ivy.erf(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
@to_ivy_arrays_and_back
def erfc(input, *, out=None):
    return 1.0 - ivy.erf(input, out=out)


@to_ivy_arrays_and_back
def sgn(input, *, out=None):
    if ivy.is_complex_dtype(input.dtype):
        input_abs = ivy.abs(input, out=out)
        # TODO wrap this in Where function after solve it's errors
        if input_abs == 0:
            return 0
        else:
            return ivy.divide(input, input_abs, out=out)
    else:
        return ivy.sign(input, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    return ivy.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
@to_ivy_arrays_and_back
def masked_fill(input, mask, value):
    return ivy.where(mask, value, input, out=input)


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
@to_ivy_arrays_and_back
def igamma(input, other, *, out=None):
    return ivy.igamma(input, x=other, out=out)


@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def lgamma(input, *, out=None):
    return ivy.lgamma(input, out=out)
