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
        x = [ivy.astype(ivy.array(i), ivy.as_ivy_dtype(dtype)) for i in x]
    return ivy.mean(x, axis=axis, keepdims=keepdims, out=out)


mean.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
