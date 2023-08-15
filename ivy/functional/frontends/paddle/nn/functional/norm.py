# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def batch_norm(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    momentum=0.9,
    epsilon=1e-05,
    data_format="NCHW",
):
    normalized, _, _ = ivy.batch_norm(
        x,
        running_mean,
        running_var,
        scale=weight,
        offset=bias,
        training=training,
        eps=epsilon,
        momentum=momentum,
        data_format=data_format,
    )

    return normalized


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return ivy.layer_norm(x, normalized_shape, weight, bias, epsilon)
