# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def mean(input, axis=None, keepdim=False, out=None):
    ret = ivy.mean(input, axis=axis, keepdims=keepdim, out=out)
    ret = ivy.expand_dims(ret, axis=-1) if ret.ndim == 0 else ret
    return ret
