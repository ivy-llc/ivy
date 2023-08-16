# local
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
import ivy.functional.frontends.tensorflow.nn as tf_nn


# should have float16 as well but sqrt doesn't support it
@to_ivy_arrays_and_back
@with_supported_dtypes({"2.13.0 and below": ("float32",)}, "tensorflow")
def fused_batch_norm(
    x,
    scale,
    offset,
    mean=None,
    variance=None,
    epsilon=1e-3,
    data_format="NHWC",
    is_training=True,
    name=None,
    exponential_avg_factor=1.0,
):
    min_epsilon = 1.001e-5
    epsilon = epsilon if epsilon > min_epsilon else min_epsilon

    dims = len(x.shape)
    if data_format[1] == "C":
        if dims == 4:
            x = ivy.permute_dims(x, axes=(0, 2, 3, 1))
        elif dims == 5:
            x = ivy.permute_dims(x, axes=(0, 2, 3, 4, 1))
        else:
            raise ivy.utils.exceptions.IvyException(
                "input tensor must be of 4 or 5 dimensions, got {}".format(dims)
            )

    scale = scale.astype(ivy.float32)
    offset = offset.astype(ivy.float32)
    old_mean = mean.astype(ivy.float32)
    old_var = variance.astype(ivy.float32)
    x = x.astype(ivy.float32)

    if is_training:
        depth = x.shape[-1]
        rest_size = ivy.prod(x.shape) // depth
        x_rest_by_depth = ivy.reshape(x, [rest_size, depth])
        mean = ivy.mean(x_rest_by_depth, axis=0, keepdims=True)
        variance = ivy.var(x_rest_by_depth, axis=0, keepdims=True)
        y = ivy.reshape(
            scale * (x_rest_by_depth - mean) / ivy.sqrt(variance + epsilon) + offset,
            x.shape,
        )
        float_rest_size = ivy.astype(rest_size, x.dtype)
        variance = (
            variance * float_rest_size / (float_rest_size - 1)
            if rest_size > 1
            else variance
        )
        mean = ivy.reshape(
            mean * exponential_avg_factor + old_mean * (1 - exponential_avg_factor),
            old_mean.shape,
        )
        variance = ivy.reshape(
            variance * exponential_avg_factor + old_var * (1 - exponential_avg_factor),
            old_var.shape,
        )
    else:
        y = scale * (x - old_mean) / ivy.sqrt(old_var + epsilon) + offset

    # permute dimensions back
    if data_format[1] == "C":
        if dims == 4:
            y = ivy.permute_dims(y, axes=(0, 3, 1, 2))
        elif dims == 5:
            y = ivy.permute_dims(y, axes=(0, 4, 1, 2, 3))

    if is_training:
        return y, mean, variance
    else:
        return y, old_mean, old_var


@with_unsupported_dtypes({"2.13.0 and below": ("float16",)}, "tensorflow")
def depthwise_conv2d(
    input,
    filter,
    strides,
    padding,
    rate=None,
    name=None,
    data_format=None,
    dilations=None,
):
    if rate:
        dilations = rate
    return tf_nn.depthwise_conv2d(
        input,
        filter,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "tensorflow",
)
def separable_conv2d(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    rate=None,
    name=None,
    data_format=None,
    dilations=None,
):
    if rate:
        dilations = rate
    return tf_nn.separable_conv2d(
        input,
        depthwise_filter,
        pointwise_filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.13.0 and below": ("float16",)},
    "tensorflow",
)
def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None, input=None):
    if input is not None and value is not None:
        raise ivy.utils.exceptions.IvyException(
            "Cannot specify both 'value' and 'input'."
        )
    return tf_nn.max_pool2d(
        input if input is not None else value,
        ksize,
        strides,
        padding,
        data_format=data_format,
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.13.0 and below": ("float32",)},
    "tensorflow",
)
def depthwise_conv2d_backprop_input(
    input_sizes,
    filter,
    out_backprop,
    strides,
    padding,
    data_format,
    dilations,
    name=None,
):
    if input_sizes is None:
        raise ivy.utils.exceptions.IvyException("Cannot specify'input_sizes'.")
    return tf_nn.depthwise_conv2d_backprop_input(
        input_sizes,
        filter,
        out_backprop,
        strides,
        padding,
        data_format,
        dilations,
        name,
    )
