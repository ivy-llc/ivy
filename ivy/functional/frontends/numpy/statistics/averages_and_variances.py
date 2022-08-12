# global
import ivy


def mean(
    x,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    *,
    where=True
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.mean(x, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


mean.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
