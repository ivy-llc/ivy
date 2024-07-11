# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, scale=weight, offset=bias, eps=epsilon)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    if axis < 0:
        axis = ivy.get_num_dims(x) + axis
    return ivy.lp_normalize(x, p=p, axis=axis)
