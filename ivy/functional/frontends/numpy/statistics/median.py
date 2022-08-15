import ivy


def median(
    a,
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
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
    ret = ivy.median(a, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


medain.unsupported_dtypes = {"torch": ("float16",)}
