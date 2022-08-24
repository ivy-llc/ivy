import ivy


def sort(
    x,
    /,
    *,
    axis=-1,
    descending=False,
    stable=True,
    out=None,
):
    return ivy.sort(x, axis=axis, descending=descending, stable=stable, out=out)
