# global
import ivy


def argsort(
    x,
    /,
    *,
    axis=-1,
    kind=None,
    order=None,
):
    return ivy.argsort(x, axis=axis)


def argmax(x,
           /,
           *,
           axis=None,
           keepdims=False,
           out=None
):
    return ivy.argmax(x, axis=axis)
