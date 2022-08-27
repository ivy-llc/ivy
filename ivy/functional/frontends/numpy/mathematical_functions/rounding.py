import ivy


def ceil(
        x,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
        signature=None,
        extobj=None
):
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))
    if out is None:
        out = ivy.empty(x.shape)
    return ivy.where(where, ivy.ceil(x, out=x), out, out=out)


ceil.unsupported_dtypes = {"torch": ("float16",)}
