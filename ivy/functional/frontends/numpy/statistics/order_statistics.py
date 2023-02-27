# global
import ivy


def percentile(a,
               q,
               /,
               *,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               interpolation=None):

    axis = tuple(axis) if isinstance(axis, list) else axis
    a = ivy.astype(ivy.array(a))
    ret = ivy.percentile(a, q, axis=axis, overwrite_input=overwrite_input, method=method, keepdims=keepdims,
                         interpolation=interpolation, out=out)

    return ret
