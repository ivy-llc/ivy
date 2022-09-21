# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def argmax(a, axis=None, out=None, *, keepdims=None):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)


def any(x, /, axis=None, out=None, keepdims=False, *, where=True):
    ret = ivy.where(ivy.array(where), ivy.array(x), ivy.zeros_like(x))
    ret = ivy.any(ret, axis=axis, keepdims=keepdims, out=out)
    return ret


def transpose(a, /, axes=None):
    return ivy.matrix_transpose(a, out=None)
