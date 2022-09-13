# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


@from_zero_dim_arrays_to_float
def argmax(a, axis=None, out=None, *, keepdims=None):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out)
