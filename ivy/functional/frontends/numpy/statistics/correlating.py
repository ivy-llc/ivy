import ivy


def sum(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    out=None,
    initial=None,
    where=None,
):
    return ivy.sum(
        x,
        axis=axis,
        dtype=dtype,
        keepdims=keepdims,
        out=out,
        initial=initial,
        where=where,
    )


sum.unsupported_dtypes = {"torch": ("float16",)}
