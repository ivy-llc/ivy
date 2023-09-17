# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, weight, bias, epsilon)


# local_response_norm
@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def local_response_norm(
    x, size, alpha=0.0001, beta=0.75, k=1.0, data_format="NCHW", name=None
):
    return ivy.local_response_norm(x, size, alpha, beta, k, data_format)
