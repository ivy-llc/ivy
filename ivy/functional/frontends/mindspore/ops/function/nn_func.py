"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def selu(input_x):
    return ivy.selu(input_x)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


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
    groups=1
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
    pad_mode='valid',
    padding=0,
    dilation=1,
    groups=1
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
    pad_mode='valid',
    padding=0,
    dilation=1,
    groups=1
    ):
    if pad_mode == "valid" or pad_mode == "same":
        padding = pad_mode
    elif pad_mode == "pad":
        padding = padding
    else:
        raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
    return _conv(input, weight, bias, stride, padding, dilation, groups)
