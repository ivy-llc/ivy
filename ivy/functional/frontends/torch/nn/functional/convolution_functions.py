# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def _valid_shapes(input, weight, bias, stride, padding, groups, transpose=False):

    in_channels = input.shape[1]
    out_channels = weight.shape[0] if not transpose else weight.shape[1] * groups

    ivy.utils.assertions.check_equal(
        in_channels % groups, 0, message="in_channels must be divisible by groups"
    )
    ivy.utils.assertions.check_equal(
        out_channels % groups, 0, message="out_channels must be divisible by groups"
    )

    if bias is not None:
        ivy.utils.assertions.check_equal(
            bias.shape[0],
            out_channels,
            message="bias must be same shape as out_channels",
        )

    if padding == "same":
        if isinstance(stride, int):
            ivy.utils.assertions.check_equal(
                stride, 1, message="padding cannot be 'same' for stride > 1"
            )
        else:
            for i in stride:
                ivy.utils.assertions.check_equal(
                    i, 1, message="padding cannot be 'same' for stride > 1"
                )

    if not transpose:
        in_channels_by_groups = weight.shape[1]
        ivy.utils.assertions.check_equal(
            in_channels,
            in_channels_by_groups * groups,
            message="in_channels must be consistent between input and weight",
        )
    else:
        ivy.utils.assertions.check_equal(
            in_channels,
            weight.shape[0],
            message="in_channels must be consistent between input and weight",
        )


def _conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    dims = len(input.shape) - 2
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        padding = [padding] * dims if isinstance(padding, int) else padding
        pad_width = [(0, 0), (0, 0), *[(p, p) for p in padding]]
        input = ivy.zero_pad(input, pad_width)
        padding = "VALID"

    weight = ivy.permute_dims(weight, axes=(*range(2, dims + 2), 1, 0))

    ret = ivy.conv(
        input,
        weight,
        stride,
        padding,
        dims=dims,
        data_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
    )
    if bias is not None:
        return ivy.add(ret, ivy.expand_dims(bias, axis=(0, *range(2, dims + 2))))
    return ret


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


# ToDo: add support for dilation > 1
# ToDo: add support for output_padding > padding
def _conv_transpose(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    dims = len(input.shape) - 2
    weight = ivy.permute_dims(weight, axes=(*range(2, dims + 2), 0, 1))
    for i in range(dims):
        weight = ivy.flip(weight, axis=i)
    padding, output_padding = map(
        lambda x: [x] * dims if isinstance(x, int) else x, [padding, output_padding]
    )
    pad_widths = [(weight.shape[i] - 1,) * 2 for i in range(dims)]
    ret = ivy.conv_general_dilated(
        input,
        weight,
        1,
        pad_widths,
        dims=dims,
        data_format="channel_first",
        feature_group_count=groups,
        x_dilations=stride,
        bias=bias,
    )
    unpad_slice = (slice(None),) * 2
    for i in range(dims):
        unpad_slice += (
            slice(padding[i], ret.shape[2 + i] - padding[i] + output_padding[i], 1),
        )
    ret = ret[unpad_slice]
    return ret


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return _conv_transpose(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return _conv_transpose(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def conv_transpose3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return _conv_transpose(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )


@to_ivy_arrays_and_back
def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    return ivy.unfold(
        input,
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )


@to_ivy_arrays_and_back
def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return ivy.fold(
        input,
        output_size,
        kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
