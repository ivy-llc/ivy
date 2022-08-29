# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def prod(
        x,
        /,
        *,
        axis=None,
        dtype=None,
        out=None,
        keepdims=None,
        initial=None,
        where=None
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    if initial:
        s = ivy.shape(x, as_array=True)
        s[axis] = 1
        header = ivy.full(ivy.Shape(s), initial)
        if where:
            x = ivy.where(where, x, ivy.default(out, ivy.zeros_like(x)))
        x = ivy.concat([x, header], axis=axis)
    ret = ivy.prod(x, axis=axis, keepdims=keepdims, out=out)
    return ret


prod.unsupported_dtypes = {"torch": ("float16",)}
