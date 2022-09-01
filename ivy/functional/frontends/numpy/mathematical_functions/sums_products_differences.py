# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def prod(
    x, /, *, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    if ivy.is_array(where):
        x = ivy.where(where, x, ivy.default(out, ivy.zeros_like(x)), out=out)
    ret = ivy.prod(x, axis=axis, keepdims=keepdims, out=out)
    return ret


prod.unsupported_dtypes = {"torch": ("float16",)}


def cumsum(a, /, axis=None, dtype=None, out=None):
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


def cumprod(a, /, axis=None, dtype=None, out=None):
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)
