# global
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def add(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.add(input, other, alpha=alpha, out=out)


@to_ivy_arrays_and_back
def tan(input, *, out=None):
    return ivy.tan(input, out=out)


@to_ivy_arrays_and_back
def atan(input, *, out=None):
    return ivy.atan(input, out=out)


@to_ivy_arrays_and_back
def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


@to_ivy_arrays_and_back
def cos(input, *, out=None):
    return ivy.cos(input, out=out)


@to_ivy_arrays_and_back
def sin(input, *, out=None):
    return ivy.sin(input, out=out)


@to_ivy_arrays_and_back
def acos(input, *, out=None):
    return ivy.acos(input, out=out)


@to_ivy_arrays_and_back
def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


@to_ivy_arrays_and_back
def acosh(input, *, out=None):
    return ivy.acosh(input, out=out)


@to_ivy_arrays_and_back
def arccosh(input, *, out=None):
    return ivy.acosh(input, out=out)


@to_ivy_arrays_and_back
def arccos(input, *, out=None):
    return ivy.acos(input, out=out)


@to_ivy_arrays_and_back
def abs(input, *, out=None):
    return ivy.abs(input, out=out)


@to_ivy_arrays_and_back
def cosh(input, *, out=None):
    return ivy.cosh(input, out=out)


@to_ivy_arrays_and_back
def subtract(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.subtract(input, other * alpha, out=out)

sub = subtract

@to_ivy_arrays_and_back
def exp(input, *, out=None):
    return ivy.exp(input, out=out)


@to_ivy_arrays_and_back
def asin(input, *, out=None):
    return ivy.asin(input, out=out)


@to_ivy_arrays_and_back
def arcsin(input, *, out=None):
    return ivy.asin(input, out=out)


@to_ivy_arrays_and_back
def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


@to_ivy_arrays_and_back
def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


@to_ivy_arrays_and_back
def arctanh(input, *, out=None):
    return ivy.atanh(input, out=out)


@to_ivy_arrays_and_back
def log2(input, *, out=None):
    return ivy.log2(input, out=out)


@to_ivy_arrays_and_back
def square(input, *, out=None):
    return ivy.square(input, out=out)


@to_ivy_arrays_and_back
def atan2(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.atan2(input, other, out=out)


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


@to_ivy_arrays_and_back
def log10(input, *, out=None):
    return ivy.log10(input, out=out)


@to_ivy_arrays_and_back
def trunc(input, *, out=None):
    return ivy.trunc(input, out=out)


@to_ivy_arrays_and_back
def sqrt(input, *, out=None):
    return ivy.sqrt(input, out=out)


@to_ivy_arrays_and_back
def sign(input, *, out=None):
    return ivy.sign(input, out=out)


@to_ivy_arrays_and_back
def absolute(input, *, out=None):
    return ivy.abs(input, out=out)


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


@to_ivy_arrays_and_back
def ceil(input, *, out=None):
    return ivy.ceil(input, out=out)


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
    return ivy.multiply(input, other, out=out)


multiply = mul


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
def flipud(input):
    return ivy.flipud(input)
