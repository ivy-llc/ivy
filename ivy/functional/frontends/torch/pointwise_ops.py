# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, integer_arrays_to_float
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def add(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.add(input, other, alpha=alpha, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def tan(input, *, out=None):
    return ivy.tan(input, out=out)


@to_ivy_arrays_and_back
def remainder(input, other, *, out=None):
    if ivy.is_array(input) and ivy.isscalar(other):
        other = ivy.full(input.shape, other)
    return ivy.remainder(input, other, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def atan(input, *, out=None):
    return ivy.atan(input, out=out)


arctan = atan


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def cos(input, *, out=None):
    return ivy.cos(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sin(input, *, out=None):
    return ivy.sin(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def acos(input, *, out=None):
    return ivy.acos(input, out=out)


arccos = acos


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
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


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def exp(input, *, out=None):
    return ivy.exp(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def asin(input, *, out=None):
    return ivy.asin(input, out=out)


arcsin = asin


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


arctanh = atanh


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log2(input, *, out=None):
    return ivy.log2(input, out=out)


@to_ivy_arrays_and_back
def square(input, *, out=None):
    return ivy.square(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def atan2(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.atan2(input, other, out=out)


arctan2 = atan2


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


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log10(input, *, out=None):
    return ivy.log10(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def trunc(input, *, out=None):
    return ivy.trunc(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def sqrt(input, *, out=None):
    return ivy.sqrt(input, out=out)


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


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def round(input, *, decimals=0, out=None):
    m = ivy.full(input.shape, 10**decimals)
    upscale = ivy.multiply(input, m, out=out)
    rounded = ivy.round(upscale, out=out)
    return ivy.divide(rounded, m, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def ceil(input, *, out=None):
    return ivy.ceil(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def clamp(input, min=None, max=None, *, out=None):
    ivy.assertions.check_all_or_any_fn(
        min,
        max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    input = ivy.array(input)
    if min is None:
        return ivy.minimum(input, max, out=out)
    if max is None:
        return ivy.maximum(input, min, out=out)
    return ivy.clip(input, min, max, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def clip(input, min=None, max=None, *, out=None):
    ivy.assertions.check_all_or_any_fn(
        min,
        max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    input = ivy.array(input)
    if min is None:
        return ivy.minimum(input, max, out=out)
    if max is None:
        return ivy.maximum(input, min, out=out)
    return ivy.clip(input, min, max, out=out)


@to_ivy_arrays_and_back
def mul(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.multiply(input, other, out=out)


multiply = mul


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def div(input, other, *, rounding_mode=None, out=None):
    if rounding_mode is not None:
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        promoted = input.dtype
        if rounding_mode == "trunc":
            return ivy.trunc_divide(input, other, out=out).astype(promoted)
        else:
            return ivy.floor_divide(input, other, out=out).astype(promoted)
    else:
        return ivy.divide(input, other, out=out)


@to_ivy_arrays_and_back
def reciprocal(input, *, out=None):
    return ivy.reciprocal(input)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def floor(input, *, out=None):
    return ivy.floor(input, out=out)


@to_ivy_arrays_and_back
def flipud(input):
    return ivy.flipud(input)


@integer_arrays_to_float
@to_ivy_arrays_and_back
def deg2rad(input, *, out=None):
    return ivy.array(input * 3.1416 / 180, out=out)


arcsinh = asinh


divide = div


@to_ivy_arrays_and_back
def true_divide(input, other, *, out=None):
    return ivy.divide(input, other, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def log1p(input, *, out=None):
    return ivy.log1p(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def addcdiv(input, tensor1, tensor2, *, value=1, out=None):
    return ivy.add(input, ivy.multiply(value, ivy.divide(tensor1, tensor2)), out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    return ivy.add(input, ivy.multiply(value, ivy.multiply(tensor1, tensor2)), out=out)


@to_ivy_arrays_and_back
def pow(input, exponent, *, out=None):
    return ivy.pow(input, exponent, out=out)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def log(input, *, out=None):
    return ivy.log(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def logaddexp(x1, x2, out=None):
    return ivy.logaddexp(x1, x2, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def exp2(input, out=None):
    return ivy.exp2(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def rsqrt(input, *, out=None):
    return ivy.reciprocal(ivy.sqrt(input), out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
@to_ivy_arrays_and_back
def expm1(input, out=None):
    return ivy.expm1(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def logaddexp2(x1, x2, out=None):
    return ivy.logaddexp2(x1, x2, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def i0(x, out=None):
    return ivy.i0(x, out=out)
