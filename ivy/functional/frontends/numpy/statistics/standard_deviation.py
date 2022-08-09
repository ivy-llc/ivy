import ivy

def std(
    x,
    axis=None,
    correction=0.0,
    keepdims=False,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    ret = ivy.std(x, axis, correction, keepdims, out=out)
    if x.size == 0:
        ret = 0
    if ivy.isnan(x):
        ret = ivy.nan
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
std.unsupported_dtypes = {"torch": ("float16",)}
