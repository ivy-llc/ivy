# local
import ivy


def where(cond, x1=None, x2=None, /):
    if x1 is None and x2 is None:
        # numpy where behaves as np.asarray(condition).nonzero() when x and y
        # not included
        return ivy.asarray(cond).nonzero()
    elif x1 is not None and x2 is not None:
        return ivy.where(cond, x1, x2)
    else:
        raise TypeError("where takes either 1 or 3 arguments")


def nonzero(a):
    return ivy.nonzero(a)


def argmin(x, /, *, axis=None, keepdims=False, out=None):
    return ivy.native_array(
        ivy.argmin(x, axis=axis, out=out, keepdims=keepdims), dtype=x.dtype
    )
