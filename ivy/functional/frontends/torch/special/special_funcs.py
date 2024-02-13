import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
)
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
@to_ivy_arrays_and_back
def erfc(input, *, out=None):
    return 1.0 - ivy.erf(input, out=out)


@with_unsupported_dtypes({"2.2 and below": ("float16", "complex", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def erfcx(input, *, out=None):
    ret = erfc(input) * ivy.exp(input**2)
    return ret


@to_ivy_arrays_and_back
def erfinv(input, *, out=None):
    return ivy.erfinv(input, out=out)
