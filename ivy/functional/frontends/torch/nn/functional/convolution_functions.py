# global
import math

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def _div_rtn(x, y):
    q = x / y
    r = x % y
    if (r != 0) and ((r < 0) != (y < 0)):
        q = q - 1
    return q


def _valid_shapes(input, weight, bias, stride, padding, groups, transpose=False):

    in_channels = input.shape[1]
    out_channels = weight.shape[0] if not transpose else weight.shape[1] * groups

    ivy.assertions.check_equal(
        in_channels % groups, 0, message="in_channels must be divisible by groups"
    )
    ivy.assertions.check_equal(
        out_channels % groups, 0, message="out_channels must be divisible by groups"
    )

    if bias is not None:
        ivy.assertions.check_equal(
            bias.shape[0],
            out_channels,
            message="bias must be same shape as out_channels",
        )

    if padding == "same":
        if isinstance(stride, int):
            ivy.assertions.check_equal(
                stride, 1, message="padding cannot be 'same' for stride > 1"
            )
        else:
            for i in stride:
                ivy.assertions.check_equal(
                    i, 1, message="padding cannot be 'same' for stride > 1"
                )

    if not transpose:
        in_channels_by_groups = weight.shape[1]
        ivy.assertions.check_equal(
            in_channels,
            in_channels_by_groups * groups,
            message="in_channels must be consistent between input and weight",
        )
    else:
        ivy.assertions.check_equal(
            in_channels,
            weight.shape[0],
            message="in_channels must be consistent between input and weight",
        )


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if type(padding) == str:
        padding = padding.upper()
    else:
        _pad_w = padding if isinstance(padding, int) else padding[0]
        input = ivy.zero_pad(
            input,
            pad_width=[(0, 0), (0, 0), (_pad_w, _pad_w)],
        )
        padding = "VALID"

    weight = ivy.permute_dims(weight, axes=(2, 1, 0))

    ret = ivy.conv(
        input,
        weight,
        stride,
        padding,
        data_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
        dims=1,
    )

    if bias is not None:
        return ivy.add(ret, ivy.expand_dims(bias, axis=(0, 2)))
    return ret


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        _pad_h, _pad_w = (
            (padding, padding) if isinstance(padding, int) else (padding[0], padding[1])
        )
        input = ivy.zero_pad(
            input, pad_width=[(0, 0), (0, 0), (_pad_h, _pad_h), (_pad_w, _pad_w)]
        )
        padding = "VALID"

    weight = ivy.permute_dims(weight, axes=(2, 3, 1, 0))
    ret = ivy.conv(
        input,
        weight,
        stride,
        padding,
        data_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
    )
    if bias is not None:
        return ivy.add(ret, ivy.expand_dims(bias, axis=(0, 2, 3)))
    return ret


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        _pad_t, _pad_h, _pad_w = (
            (padding, padding, padding)
            if isinstance(padding, int)
            else (padding[0], padding[1], padding[2])
        )
        input = ivy.zero_pad(
            input,
            pad_width=[
                (0, 0),
                (0, 0),
                (_pad_t, _pad_t),
                (_pad_h, _pad_h),
                (_pad_w, _pad_w),
            ],
        )
        padding = "VALID"

    weight = ivy.permute_dims(weight, axes=(2, 3, 4, 1, 0))
    ret = ivy.conv(
        input,
        weight,
        stride,
        padding,
        data_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
        dims=3,
    )
    if bias is not None:
        return ivy.add(ret, ivy.expand_dims(bias, axis=(0, 2, 3, 4)))
    return ret


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    _valid_shapes(input, weight, bias, stride, padding, groups, transpose=True)

    if type(padding) == str:
        padding = padding.upper()
    else:
        _pad_w = padding if isinstance(padding, int) else padding[0]
        padding = [(_pad_w, _pad_w)]

    weight = ivy.permute_dims(weight, axes=(2, 0, 1))

    ret = ivy.conv_general_transpose(
        input,
        weight,
        stride,
        padding,
        dims=1,
        data_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
    )

    if bias is not None:
        ret = ivy.add(ret, ivy.expand_dims(bias, axis=(0, 2)))

    _out_pad_w = output_padding if isinstance(output_padding, int) else output_padding[0]
    ret = ivy.zero_pad(
        ret,
        pad_width=[(0, 0), (0, 0), (_out_pad_w,) * 2],
    )
    return ret


@to_ivy_arrays_and_back
def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    if input.ndim != 4:
        raise ivy.exceptions.IvyException("only batched 4D inputs are supported")
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_shape = [
        (input.shape[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        for i in range(2)
    ]
    ret = ivy.zeros((*input.shape[0:2], *kernel_size, *output_shape), dtype=input.dtype)
    input_padded = ivy.zero_pad(
        input,
        ((0, 0),) * 2 + ((padding[0],) * 2, (padding[1],) * 2),
    )
    ret = ret.to_numpy()
    input_padded = input_padded.to_numpy()
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            ret[:, :, :, :, i, j] = input_padded[
                                       :,
                                       :,
                                       i_in:i_in+kernel_size[0]*dilation[0]:dilation[0],
                                       j_in:j_in+kernel_size[1]*dilation[1]:dilation[1],
                                       ]
    return ivy.reshape(ret, (input.shape[0], input.shape[1]*math.prod(kernel_size), -1))


@to_ivy_arrays_and_back
def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    orig_ndim = input.ndim
    if orig_ndim == 2:
        input = ivy.expand_dims(input, axis=0)
    elif orig_ndim != 3:
        raise ivy.exceptions.IvyException("only 2D or batched 3D inputs are supported")
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_size = [output_size] * 2 if isinstance(output_size, int) else output_size
    x_shape = [
        (output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        for i in range(2)
    ]
    n_channels = input.shape[1] // math.prod(kernel_size)
    input = ivy.reshape(input, (input.shape[0], n_channels, *kernel_size, *x_shape))
    input_padded = ivy.zero_pad(
        input,
        ((0, 0),) * 4 + ((padding[0],) * 2, (padding[1],) * 2),
    )
    output = ivy.zeros((input.shape[0], n_channels, *output_size), dtype=input.dtype)
    output = output.to_numpy()
    input_padded = input_padded.to_numpy()
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            h_start = i * stride[0] * dilation[0]
            h_end = h_start + kernel_size[0] * dilation[0]
            w_start = j * stride[1] * dilation[1]
            w_end = w_start + kernel_size[1] * dilation[1]
            sub_matrix = input_padded[
                         :, :, :, :, h_start:h_end:dilation[0], w_start:w_end:dilation[1]
                         ]
            output[:, :, i, j] = ivy.sum(sub_matrix, axis=(2, 3, 4, 5))
    return ivy.array(output) if orig_ndim == 3 else ivy.squeeze(output, axis=0)
