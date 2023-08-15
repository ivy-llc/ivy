"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


def _broadcast_pooling_helper(x, pool_dims: str = "2d", name: str = "padding"):
    dims = {"1d": 1, "2d": 2, "3d": 3}

    if isinstance(x, int):
        return tuple([x for _ in range(dims[pool_dims])])

    if len(x) == 1:
        return tuple([x[0] for _ in range(dims[pool_dims])])
    elif len(x) == dims[pool_dims]:
        return tuple(x)
    elif len(x) != dims[pool_dims]:
        raise ValueError(
            f"`{name}` must either be a single int, "
            f"or a tuple of {dims[pool_dims]} ints. "
        )


@with_supported_dtypes(
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def dropout2d(input, p=0.5, training=True):
    return ivy.dropout2d(input, p, training=training, data_format="NCHW")


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def selu(input_x):
    return ivy.selu(input_x)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def log_softmax(input, axis=-1):
    return ivy.log_softmax(input)


def _valid_shapes(input, weight, bias, stride, padding, groups, transpose=False):
    in_channels = input.shape[1]
    out_channels = weight.shape[0] if not transpose else weight.shape[1] * groups

    ivy.utils.assertions.check_equal(
        in_channels % groups,
        0,
        message="in_channels must be divisible by groups",
        as_array=False,
    )
    ivy.utils.assertions.check_equal(
        out_channels % groups,
        0,
        message="out_channels must be divisible by groups",
        as_array=False,
    )

    if bias is not None:
        ivy.utils.assertions.check_equal(
            bias.shape[0],
            out_channels,
            message="bias must be same shape as out_channels",
            as_array=False,
        )

    if padding == "same":
        if isinstance(stride, int):
            ivy.utils.assertions.check_equal(
                stride,
                1,
                message="padding cannot be 'same' for stride > 1",
                as_array=False,
            )
        else:
            for i in stride:
                ivy.utils.assertions.check_equal(
                    i,
                    1,
                    message="padding cannot be 'same' for stride > 1",
                    as_array=False,
                )

    if not transpose:
        in_channels_by_groups = weight.shape[1]
        ivy.utils.assertions.check_equal(
            in_channels,
            in_channels_by_groups * groups,
            message="in_channels must be consistent between input and weight",
            as_array=False,
        )
    else:
        ivy.utils.assertions.check_equal(
            in_channels,
            weight.shape[0],
            message="in_channels must be consistent between input and weight",
            as_array=False,
        )


def _conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    dims = len(input.shape) - 2
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        if isinstance(padding, int):
            padding = [*[(padding, padding) for _ in range(dims)]]
        else:
            padding = [*[(p, p) for p in padding]]

    ret = ivy.conv(
        input,
        weight,
        stride,
        padding,
        dims=dims,
        data_format="channel_first",
        filter_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
    )
    if bias is not None:
        return ivy.add(ret, ivy.expand_dims(bias, axis=(0, *range(2, dims + 2))))
    return ret


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    pad_mode="valid",
    padding=0,
    dilation=1,
    groups=1,
):
    if pad_mode == "valid" or pad_mode == "same":
        padding = pad_mode
    elif pad_mode == "pad":
        padding = padding
    else:
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
    return _conv(input, weight, bias, stride, padding, dilation, groups)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def conv1d(
    input,
    weight,
    bias=None,
    stride=1,
    pad_mode="valid",
    padding=0,
    dilation=1,
    groups=1,
):
    if pad_mode == "valid" or pad_mode == "same":
        padding = pad_mode
    elif pad_mode == "pad":
        padding = padding
    else:
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
    return _conv(input, weight, bias, stride, padding, dilation, groups)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def conv3d(
    input,
    weight,
    bias=None,
    stride=1,
    pad_mode="valid",
    padding=0,
    dilation=1,
    groups=1,
):
    if pad_mode == "valid" or pad_mode == "same":
        padding = pad_mode
    elif pad_mode == "pad":
        padding = padding
    else:
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
    return _conv(input, weight, bias, stride, padding, dilation, groups)


def kl_div(logits, labels, reduction="mean"):
    """
    Computes the Kullback-Leibler (KL) Divergence between the logits and the labels.

    Parameters:
        logits (numpy array): The input logits array.
        labels (numpy array): The label array which has the same shape as logits.
        reduction (str): Specifies the reduction to be applied to the output.
                         Its value must be one of 'none', 'mean', 'batchmean',
                         or 'sum'. Default: 'mean'.

    Returns:
        float or numpy array: If reduction is 'none', then output is
        a numpy array and has the same shape as logits.
                              Otherwise, it is a scalar (float).
    """
    assert ivy.shape(logits) == ivy.shape(
        labels
    ), "logits and labels must have the same shape."
    L = labels * (ivy.log(labels) - logits)
    if reduction == "none":
        return L
    elif reduction == "mean":
        return ivy.mean(L)
    elif reduction == "batchmean":
        return ivy.mean(L, axis=0)
    elif reduction == "sum":
        return ivy.sum(L)
    else:
        raise ValueError(
            "Invalid reduction mode. Supported values are 'none', 'mean', 'batchmean',"
            " or 'sum'."
        )


@with_supported_dtypes(
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def dropout3d(input, p=0.5, training=True):
    return ivy.dropout3d(input, p, training=training, data_format="NCDHW")


@with_supported_dtypes(
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=False,
    recompute_scale_factor=False,
):
    return ivy.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )


@with_supported_dtypes(
    {
        "2.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def pad(input, pad_width, mode="constant", constant_values=0):
    return ivy.pad(input, pad_width, mode=mode, constant_values=constant_values)


@with_supported_dtypes(
    {"2.0.0 and below": ("float16", "float32", "float64")}, "mindspore"
)
@to_ivy_arrays_and_back
def adaptive_avg_pool2d(input, output_size):
    return ivy.adaptive_avg_pool2d(input, output_size)


@to_ivy_arrays_and_back
def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    pad_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # Figure out input dims N
    input_rank = input.ndim

    if input_rank == 4:
        # NCHW
        data_format = "NCHW"

    kernel_size = _broadcast_pooling_helper(kernel_size, "2d", name="kernel_size")
    stride = _broadcast_pooling_helper(stride, "2d", name="stride")
    padding = _broadcast_pooling_helper(padding, "2d", name="padding")
    kernel_pads = list(zip(kernel_size, padding))

    # Padding should be less than or equal to half of kernel size
    if not all([pad <= kernel / 2 for kernel, pad in kernel_pads]):
        raise ValueError(
            "pad should be smaller than or equal to half of kernel size, "
            f"but got padding={padding}, kernel_size={kernel_size}. "
        )

    # Figure out padding string
    if all([pad == ivy.ceil((kernel - 1) / 2) for kernel, pad in kernel_pads]):
        padding_str = "SAME"
    else:
        padding_str = "VALID"

    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding_str,
        data_format=data_format,
        pad_mode=pad_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@to_ivy_arrays_and_back
def bias_add(input_x, bias):
    if not isinstance(input_x, ivy.array) or not isinstance(bias, ivy.array):
        raise TypeError("Both input_x and bias must be  arrays")

    if input_x.ndim < 2 or input_x.ndim > 5:
        raise TypeError("Dimension of input_x must be in the range [2, 5]")

    if bias.shape != (input_x.shape[-1],):
        raise ValueError("Shape of bias must match the channel dimension of input_x")

    broadcasted_bias = ivy.broadcast_to(bias, input_x.shape)
    output = input_x + broadcasted_bias
    return output


@to_ivy_arrays_and_back
def flatten(input, order="C", *, start_dim=1, end_dim=-1):
    return ivy.flatten(input, order=order, start_dim=start_dim, end_dim=end_dim)


@with_supported_dtypes(
    {"2.0.0 and below": ("float16", "float32", "float64")},
    "mindspore",
)
@to_ivy_arrays_and_back
def fast_gelu(input_x):
    return (input_x / (1 + ivy.exp(-1.702 * ivy.abs(input_x)))) * ivy.exp(
        0.851 * (input_x - ivy.abs(input_x))
    )


@with_supported_dtypes({"2.0.0 and below": ("float32", "float64")}, "mindspore")
@to_ivy_arrays_and_back
def softshrink(x, lambd=0.5):
    low = ivy.where(ivy.less(input, -lambd), ivy.add(input, lambd), 0)
    up = ivy.where(ivy.greater(input, lambd), ivy.subtract(input, lambd), 0)
    return ivy.add(low, up)


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    gumbels = -ivy.empty_like(logits).exponential().log()
    gumbels = (logits + gumbels) / tau
    y_soft = ivy.softmax(gumbels, axis=dim)

    if hard:
        indices = y_soft.max(axis=dim, keepdims=True)[1]
        y_hard = ivy.zeros_like(logits)
        updates = ivy.ones_like(indices)
        y_hard = ivy.scatter_nd(indices, updates, reduction="replace", out=y_hard)

        ret = y_hard - y_soft.stop_gradient(preserve_type=True) + y_soft
    else:
        ret = y_soft

    return ret
