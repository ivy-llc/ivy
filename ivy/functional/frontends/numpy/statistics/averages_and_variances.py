# global
import ivy


def mean(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    *,
    where=True
):
    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
    ret = ivy.mean(a, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


mean.unsupported_dtypes = {"torch": ("float16",)}
