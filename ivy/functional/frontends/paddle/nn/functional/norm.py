# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, weight, bias, epsilon)



@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
def batch_norm(x, gamma, beta, moving_mean, moving_var, epsilon=1e-5):
    return ivy.batch_norm(x, gamma, beta, moving_mean, moving_var, epsilon)