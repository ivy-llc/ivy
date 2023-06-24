# local
import ivy


def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None
):
    axis = tuple(axis) if isinstance(axis, list) else axis

    ret = ivy.quantile(a, axis=axis, keepdims=keepdims, out=out)
    return ret
