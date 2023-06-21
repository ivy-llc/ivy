# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
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
    ivy.utils.assertions.check_less(
        ivy.get_num_dims(x),
        2,
        message="input dim must be larger than 1",
        as_array=False,
    )

    n_dims = len(x.shape)
    input = ivy.permute_dims(x, axes=(0, *range(2, n_dims), 1))

    normalized, _, _ = ivy.batch_norm(
        x=input,
        mean=running_mean,
        variance=running_var,
        scale=weight,
        offset=bias,
        training=training,
        eps=epsilon,
        momentum=momentum,
        data_format=data_format,
    )

    return ivy.permute_dims(normalized, axes=(0, n_dims - 1, *range(1, n_dims - 1)))
