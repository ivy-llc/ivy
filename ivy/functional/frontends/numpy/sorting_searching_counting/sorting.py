import ivy

def argsort(
    x,
    /,
    *,
    axis=-1,
    descending=False,
    stable=True,
    out=None,
):
    return ivy.argsort(x, axis=axis, descending=descending, stable=stable, out=out)