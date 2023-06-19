import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
# TODO torch inplace updates running_mean and running_var
@to_ivy_arrays_and_back
def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    # tranpose the input from N,C,*S to N,*S, C
    ndims = len(input.shape)
    input = ivy.permute_dims(input, axes=(0, *range(2, ndims), 1))
    normalized, running_mean, running_var = ivy.batch_norm(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    normalized = ivy.permute_dims(normalized, axes=(0, ndims - 1, *range(1, ndims - 1)))
    return normalized


# TODO torch inplace updates running_mean and running_var
@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def instance_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    use_input_stats=False,
    momentum=0.1,
    eps=1e-5,
):
    # tranpose the input from N,C,*S to N,*S, C
    ndims = len(input.shape)
    input = ivy.permute_dims(input, axes=(0, *range(2, ndims), 1))
    normalized, running_mean, running_var = ivy.instance_norm(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=use_input_stats,
        eps=eps,
        momentum=momentum,
    )
    normalized = ivy.permute_dims(normalized, axes=(0, ndims - 1, *range(1, ndims - 1)))
    return normalized


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    if isinstance(normalized_shape, int) and normalized_shape == shape[-1]:
        axis = [-1]
    else:
        assert normalized_shape == shape[-len(normalized_shape) :]
        axis = list(range(len(shape) - len(normalized_shape), len(shape)))
    return ivy.layer_norm(input, axis, scale=weight, offset=bias, eps=eps)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    assert shape[1] % num_groups == 0
    groups = shape[1] // num_groups
    num_dims = ivy.get_num_dims(input)
    expand_dims = (
        [0, *range(2, num_dims)] if weight is not None and num_dims > 2 else [0]
    )
    ret = ivy.concat(
        [
            ivy.layer_norm(
                input[:, i * groups : (i + 1) * groups, ...],
                list(range(1, num_dims)),
                scale=(
                    ivy.expand_dims(
                        weight[i * groups : (i + 1) * groups], axis=expand_dims
                    )
                    if weight is not None
                    else None
                ),
                offset=(
                    ivy.expand_dims(
                        bias[i * groups : (i + 1) * groups], axis=expand_dims
                    )
                    if bias is not None
                    else None
                ),
                eps=eps,
            )
            for i in range(num_groups)
        ],
        axis=1,
    )

    return ret
