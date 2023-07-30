# global
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.tensorflow import check_tensorflow_casting


def _reduce_strides_dilations(dim, stride, dilations):
    if not isinstance(stride, int):
        if len(stride) > dim:
            stride = stride[1:-1]
        if len(stride) == 1 and dim != 1:
            stride = stride[0]
    if not isinstance(dilations, int):
        if len(dilations) > dim:
            dilations = dilations[1:-1]
        if len(dilations) == 1 and dim != 1:
            dilations = dilations[0]
    return stride, dilations


@to_ivy_arrays_and_back
def atrous_conv2d(value, filters, rate, padding):
    return ivy.conv2d(value, filters, 1, padding, dilations=[rate] * 2)


@to_ivy_arrays_and_back
def atrous_conv2d_transpose(value, filters, output_shape, rate, padding):
    filters = filters.swapaxes(-2, -1)
    return ivy.conv2d_transpose(
        value, filters, 1, padding, output_shape=output_shape, dilations=[rate] * 2
    )


@to_ivy_arrays_and_back
def conv1d(
    input, filters, stride, padding, data_format="NWC", dilations=None, name=None
):
    dilations = 1 if dilations is None else dilations
    stride, dilations = _reduce_strides_dilations(1, stride, dilations)
    return ivy.conv1d(
        input, filters, stride, padding, data_format=data_format, dilations=dilations
    )


@to_ivy_arrays_and_back
def conv1d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NWC",
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(1, strides, dilations)
    filters = filters.swapaxes(-2, -1)
    return ivy.conv1d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
def gelu(features, approximate=False, name=None):
    return ivy.gelu(features, approximate=approximate)


@to_ivy_arrays_and_back
def conv2d(
    input, filters, strides, padding, data_format="NHWC", dilations=None, name=None
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    return ivy.conv2d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


@to_ivy_arrays_and_back
def conv2d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NHWC",
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    filters = filters.swapaxes(-2, -1)
    return ivy.conv2d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@to_ivy_arrays_and_back
def conv3d(
    input, filters, strides, padding, data_format="NDHWC", dilations=None, name=None
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(3, strides, dilations)
    return ivy.conv3d(
        input, filters, strides, padding, data_format=data_format, dilations=dilations
    )


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def conv3d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding="SAME",
    data_format="NDHWC",
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(3, strides, dilations)
    filters = filters.swapaxes(-2, -1)
    return ivy.conv3d_transpose(
        input,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def depthwise_conv2d(
    input,
    filter,
    strides,
    padding="SAME",
    data_format="NHWC",
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    fc = filter.shape[-2]
    filter = filter.reshape(
        [*filter.shape[0:2], 1, filter.shape[-2] * filter.shape[-1]]
    )
    return ivy.conv_general_dilated(
        input,
        filter,
        strides,
        padding,
        data_format="channel_last" if data_format[-1] == "C" else "channel_first",
        dilations=dilations,
        feature_group_count=fc,
    )


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def separable_conv2d(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
    name=None,
):
    dilations = 1 if dilations is None else dilations
    strides, dilations = _reduce_strides_dilations(2, strides, dilations)
    ret = depthwise_conv2d(
        input,
        depthwise_filter,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format=data_format,
    )
    return conv2d(ret, pointwise_filter, 1, "SAME", data_format=data_format)


@to_ivy_arrays_and_back
def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
    xnormalized, _, _ = ivy.batch_norm(
        x,
        mean,
        variance,
        offset=offset,
        scale=scale,
        eps=variance_epsilon,
    )
    return xnormalized


@to_ivy_arrays_and_back
def dropout(x, rate, noise_shape=None, seed=None, name=None):
    return ivy.dropout(x, rate, noise_shape=noise_shape, seed=seed)


@to_ivy_arrays_and_back
def silu(features, beta: float = 1.0):
    beta = ivy.astype(ivy.array(beta), ivy.dtype(features))
    return ivy.multiply(features, ivy.sigmoid(ivy.multiply(beta, features)))


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    ivy.utils.assertions.check_shape(labels, logits)
    zeros = ivy.zeros_like(logits)
    max_logits = ivy.where(logits >= zeros, logits, zeros)
    neg_abs_logits = ivy.negative(ivy.abs(logits))
    neg_multiple = ivy.negative(ivy.multiply(logits, labels))
    ret_val = ivy.add(max_logits, neg_multiple)
    return ivy.add(ret_val, ivy.log1p(ivy.exp(neg_abs_logits)))


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def weighted_cross_entropy_with_logits(
    labels=None, logits=None, pos_weight=1.0, name=None
):
    ivy.utils.assertions.check_shape(labels, logits)
    ones = ivy.ones_like(labels)
    zeros = ivy.zeros_like(logits)
    log_weight = ivy.add(ones, ivy.multiply(pos_weight - 1, labels))
    ones_minus_labels = ivy.subtract(ones, labels)
    first_term = ivy.multiply(ones_minus_labels, logits)

    max_neg_logits = ivy.where(
        ivy.negative(logits) >= zeros, ivy.negative(logits), zeros
    )
    neg_abs_logits = ivy.negative(ivy.abs(logits))
    log_neg_abs_logits = ivy.log1p(ivy.exp(neg_abs_logits))
    second_term = ivy.multiply(log_weight, ivy.add(log_neg_abs_logits, max_neg_logits))
    return ivy.add(first_term, second_term)


@with_unsupported_dtypes({"2.13.0 and below": ("bfloat16",)}, "tensorflow")
@to_ivy_arrays_and_back
def local_response_normalization(
    input, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None
):
    input_shape = ivy.shape(input)
    depth = input_shape[-1]
    ivy.utils.assertions.check_equal(
        ivy.get_num_dims(input),
        4,
        message="4D input, but got input with sizes " + str(input_shape),
        as_array=False,
    )
    sqr_sum = ivy.empty(input_shape[:-1] + (0,), dtype=ivy.dtype(input))
    for d in range(depth):
        start = max(0, d - depth_radius)
        end = min(d + depth_radius + 1, depth)
        inter_channel_sum = ivy.sum(
            input[:, :, :, start:end] ** 2, axis=3, keepdims=True
        )
        sqr_sum = ivy.concat([sqr_sum, inter_channel_sum], axis=-1)
    return ivy.divide(input, ivy.pow(ivy.add(sqr_sum * alpha, bias), beta))


@to_ivy_arrays_and_back
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    return ivy.max_pool1d(input, ksize, strides, padding, data_format=data_format)


@to_ivy_arrays_and_back
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    return ivy.max_pool2d(input, ksize, strides, padding, data_format=data_format)


@to_ivy_arrays_and_back
def moments(x, axes, shift=None, keepdims=False, name=None):
    return ivy.mean(x, axis=ivy.to_list(axes), keepdims=keepdims), ivy.var(
        x, axis=ivy.to_list(axes), keepdims=keepdims
    )


@with_unsupported_dtypes(
    {
        "2.13.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def normalize_moments(counts, mean_ss, variance_ss, shift=None, name=None):
    divisor = ivy.reciprocal(counts)
    if shift is not None:
        shifted_mean = ivy.multiply(mean_ss, divisor)
        mean = ivy.add(shifted_mean, shift)
    else:
        shifted_mean = ivy.multiply(mean_ss, divisor)
        mean = shifted_mean

    variance = ivy.subtract(
        ivy.multiply(variance_ss, divisor), ivy.square(shifted_mean)
    )
    return mean, variance


@to_ivy_arrays_and_back
def bias_add(value, bias, data_format=None, name=None):
    if data_format is None:
        data_format = "N...C"

    chanel_index = data_format.find("C")
    if chanel_index != 1:
        return ivy.add(value, bias)
    else:
        value = ivy.swapaxes(value, 1, -1)
        res = ivy.add(value, bias)
        return ivy.swapaxes(res, 1, -1)


def _convolution_broadcast_helper(
    arg, num_spatial_dims, channel_index, name="dilations"
):
    # Helper to broadcast dilations and strides to correct dims
    if arg is None:
        return [1] * num_spatial_dims
    else:
        if isinstance(arg, int):
            arg = [arg]
        else:
            arg = list(arg)
        len_arg = len(arg)

        if len_arg == num_spatial_dims + 2:
            return arg

        # Broadcast to rcorrect dimensions
        if len_arg == 1:
            arg = arg * num_spatial_dims
        elif len_arg != num_spatial_dims:
            raise ValueError(
                f"{name} should be of length 1, "
                f"{num_spatial_dims} or {num_spatial_dims + 2}. "
                f"Received: {name}={arg} of length {len_arg}."
            )

    # Add dimensions for batch and channel
    if channel_index == 1:
        return [1, 1] + arg
    else:
        return [1] + arg + [1]


@to_ivy_arrays_and_back
def convolution(
    input,
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
):
    num_spatial_dims = input.ndim - 2
    if data_format is None or not data_format.startswith("NC"):
        data_format = "channel_last"
    else:
        data_format = "channel_first"

    channel_index = -1 if data_format == "channel_last" else 1
    input_depth = ivy.shape(input)[channel_index]
    filters_depth = ivy.shape(filters)[-2]

    feature_group_count = 1
    if input_depth != filters_depth:
        if input_depth % filters_depth != 0:
            raise ValueError(
                "input depth must be evenly divisible by filter depth: "
                f"{input_depth} vs {filters_depth}"
            )
        feature_group_count = input_depth // filters_depth
    return ivy.conv_general_dilated(
        input,
        filters,
        strides,
        padding,
        dims=num_spatial_dims,
        data_format=data_format,
        dilations=dilations,
        feature_group_count=feature_group_count,
    )


@with_unsupported_dtypes({"2.11.1 and below": ("complex",)}, "tensorflow")
@to_ivy_arrays_and_back
def embedding_lookup(params, ids, max_norm=None, name=None):
    return ivy.embedding(params, ids, max_norm=max_norm)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, "tensorflow")
@to_ivy_arrays_and_back
def relu(features, name=None):
    return ivy.relu(features)


@to_ivy_arrays_and_back
def relu6(features, name=None):
    return ivy.relu6(features)


@with_unsupported_dtypes({"2.13.0 and below": ("float16",)}, "tensorflow")
@to_ivy_arrays_and_back
def softmax(logits, axis=None, name=None):
    return ivy.softmax(logits, axis=axis)


@with_unsupported_dtypes({"2.13.0 and below": "float16"}, "tensorflow")
@to_ivy_arrays_and_back
def leaky_relu(features, alpha=0.2, name=None):
    return ivy.leaky_relu(features, alpha=alpha)


@to_ivy_arrays_and_back
def crelu(features, axis=-1, name=None):
    c = ivy.concat([features, -features], axis=axis)
    return ivy.relu(c)


@to_ivy_arrays_and_back
def avg_pool(input, ksize, strides, padding, data_format="NWC", name=None):
    if len(ivy.shape(input)) == 3:
        return ivy.avg_pool1d(input, ksize, strides, padding, data_format=data_format)
    elif len(ivy.shape(input)) == 4:
        return ivy.avg_pool2d(input, ksize, strides, padding, data_format=data_format)
    return ivy.avg_pool3d(input, ksize, strides, padding, data_format=data_format)


# avg_pool3d
@to_ivy_arrays_and_back
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    return ivy.avg_pool3d(input, ksize, strides, padding, data_format=data_format)


# avg_pool1d
@to_ivy_arrays_and_back
def avg_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    return ivy.avg_pool1d(input, ksize, strides, padding, data_format=data_format)


# pool
@to_ivy_arrays_and_back
def pool(
    input,
    window_shape,
    pooling_type,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
):
    return ivy.pool(
        input,
        window_shape,
        pooling_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )


# sufficient_statistics
@to_ivy_arrays_and_back
def sufficient_statistics(x, axes, shift=None, keepdims=False, name=None):
    count = 1
    shape = ivy.shape(x)
    axes = list(set(axes))
    for a in axes:
        if ivy.to_scalar(a) < 0:
            index = x.ndim + ivy.to_scalar(a)
        else:
            index = ivy.to_scalar(a)
        count *= shape[index]
    count = ivy.array(count, dtype=ivy.dtype(x))
    if shift is None:
        sum_of_elements = ivy.sum(x, axis=axes, keepdims=keepdims)
        sum_of_squares = ivy.sum(ivy.square(x), axis=axes, keepdims=keepdims)
    else:
        sum_of_elements = ivy.sum(
            (ivy.subtract(x, shift)), axis=axes, keepdims=keepdims
        )
        sum_of_squares = ivy.sum(
            (ivy.square(ivy.subtract(x, shift))), axis=axes, keepdims=keepdims
        )
        if shift.ndim == 0:
            ivy.reshape(shift, ())

    if count.ndim == 0:
        ivy.reshape(count, ())
    if sum_of_elements.ndim == 0:
        ivy.reshape(sum_of_elements, ())
    if sum_of_squares.ndim == 0:
        ivy.reshape(sum_of_squares, ())
    return count, sum_of_elements, sum_of_squares, shift


# log_poisson_loss
@to_ivy_arrays_and_back
def log_poisson_loss(targets, log_input, compute_full_loss=False, name=None):
    return ivy.log_poisson_loss(targets, log_input, compute_full_loss=compute_full_loss)


# ctc_unique_labels
@to_ivy_arrays_and_back
def ctc_unique_labels(labels, name=None):
    ctc_labels = ivy.unique_all(labels, by_value=False)
    unique_pad = ivy.pad(
        ctc_labels[0], (0, labels.size - ctc_labels[0].size), mode="constant"
    )
    return unique_pad, ctc_labels[2]


# weighted_moments
@to_ivy_arrays_and_back
def weighted_moments(x, axes, frequency_weights, keepdims=False, name=None):
    fw_x_prod = frequency_weights * x
    fw_x_prod = ivy.array(fw_x_prod)
    weighted_input_sum = ivy.sum(fw_x_prod, axis=axes, keepdims=True).astype(
        fw_x_prod.dtype
    )

    broadcasted_weights = frequency_weights + ivy.zeros_like(x)
    broadcasted_weights = ivy.array(broadcasted_weights)
    sum_of_weights = ivy.sum(broadcasted_weights, axis=axes, keepdims=True).astype(
        broadcasted_weights.dtype
    )

    divisor = ivy.reciprocal(sum_of_weights)

    weighted_input_sum, divisor = check_tensorflow_casting(weighted_input_sum, divisor)
    weighted_mean = ivy.multiply(weighted_input_sum, divisor)

    x, weighted_mean = check_tensorflow_casting(x, weighted_mean)
    squared_difference = ivy.square(ivy.subtract(x, weighted_mean))
    if isinstance(squared_difference, complex):
        squared_difference = squared_difference.real - squared_difference.imag * 1j

    fw_sq_diff_prod = frequency_weights * squared_difference
    fw_sq_diff_prod = ivy.array(fw_sq_diff_prod)
    weighted_distsq = ivy.sum(fw_sq_diff_prod, axis=axes, keepdims=True).astype(
        fw_sq_diff_prod.dtype
    )

    weighted_distsq, divisor = check_tensorflow_casting(weighted_distsq, divisor)
    weighted_variance = ivy.multiply(weighted_distsq, divisor)

    if not keepdims:
        weighted_mean = ivy.squeeze(weighted_mean, axis=axes)
        weighted_variance = ivy.squeeze(weighted_variance, axis=axes)
    return weighted_mean, weighted_variance

@to_ivy_arrays_and_back
def dropout(x, rate, noise_shape=None, seed=None, name=None):
    return ivy.dropout(x, rate, noise_shape=noise_shape, seed=seed)