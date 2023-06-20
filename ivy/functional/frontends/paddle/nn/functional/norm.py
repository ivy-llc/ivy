# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.0.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    "paddle",
)
def batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    epsilon,
    data_format,
):
    ivy.utils.assertions.check_equal(
        ivy.get_num_dims(input),
        2,
        message="input dim must be greater than 1",
        as_array=False,
    )
    n_dims = len(input.shape)
    input = ivy.permute_dims(input, axes=(0, *range(2, n_dims), 1))

    normalized = ivy.batch_norm(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=training,
        eps=epsilon,
        momentum=momentum,
        data_format=data_format,
    )[0]

    return ivy.permute_dims(normalized, axes=(0, n_dims - 1, *range(1, n_dims - 1)))
