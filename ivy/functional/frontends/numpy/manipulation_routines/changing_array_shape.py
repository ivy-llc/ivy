# local
import ivy


def reshape(x,
            /,
            shape,
            *,
            out=None,
            copy=None):
    return ivy.reshape(x, shape, copy=copy, out=out)
