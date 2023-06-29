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

    ret = ivy.quantile(
        a,
        q,
        axis=axis,
        keepdims=keepdims,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        interpolation=interpolation,
    )
    return ret
