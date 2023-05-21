# global
import math

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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
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
    if input.ndim != 4:
        raise ivy.utils.exceptions.IvyException("only batched 4D inputs are supported")
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_shape = [
        (input.shape[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        // stride[i]
        + 1
        for i in range(2)
    ]
    ret = ivy.zeros((*input.shape[0:2], *kernel_size, *output_shape), dtype=input.dtype)
    input_padded = ivy.zero_pad(
        input,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
    )
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            ret[:, :, :, :, i, j] = input_padded[
                :,
                :,
                i_in : i_in + kernel_size[0] * dilation[0] : dilation[0],
                j_in : j_in + kernel_size[1] * dilation[1] : dilation[1],
            ]
    return ivy.reshape(
        ret, (input.shape[0], input.shape[1] * math.prod(kernel_size), -1)
    )


@to_ivy_arrays_and_back
def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    orig_ndim = input.ndim
    if orig_ndim == 2:
        input = ivy.expand_dims(input, axis=0)
    elif orig_ndim != 3:
        raise ivy.utils.exceptions.IvyException(
            "only 2D or batched 3D inputs are supported"
        )
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_size = [output_size] * 2 if isinstance(output_size, int) else output_size
    input_shape = [
        (output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        // stride[i]
        + 1
        for i in range(2)
    ]
    n_batches = input.shape[0]
    n_channels = input.shape[1] // math.prod(kernel_size)
    output = ivy.zeros((n_batches, n_channels, *output_size), dtype=input.dtype)
    output_padded = ivy.zero_pad(
        output,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
    )
    k = 0
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            patch = input[:, :, k].reshape((n_batches, n_channels, *kernel_size))
            output_padded[
                :,
                :,
                i_in : i_in + kernel_size[0] * dilation[0] : dilation[0],
                j_in : j_in + kernel_size[1] * dilation[1] : dilation[1],
            ] += patch
            k += 1
    return ivy.array(
        output_padded[:, :, padding[0] : -padding[0], padding[1] : -padding[1]]
    )
