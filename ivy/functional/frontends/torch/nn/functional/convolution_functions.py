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
            for i in padding:
                ivy.assertions.check_equal(
                    i, 1, message="padding cannot be 'same' for stride > 1"
                )

    if not transpose:
        in_channels_by_groups = weight.shape[1]
        ivy.assertions.check_equal(
            in_channels,
            in_channels_by_groups * groups,
            message="in_channels must be consistent",
        )
    else:
        ivy.assertions.check_equal(
            in_channels, weight.shape[0], message="out_channels must be consistent"
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
            "uint8",
            "integer",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def unfold(input, kernel_size, dilation=1, padding=0, stride=1):

    kernel_size = ivy.repeat(ivy.asarray(kernel_size), 2)[:2]
    dilation = ivy.repeat(ivy.asarray(dilation), 2)[:2]
    padding = ivy.repeat(ivy.asarray(padding), 2)[:2]
    stride = ivy.repeat(ivy.asarray(stride), 2)[:2]

    kernel_height, kernel_width = kernel_size
    dilation_height, dilation_width = dilation
    pad_height, pad_width = padding
    stride_height, stride_width = stride

    ivy.assertions.check_true(
        kernel_width > 0 and kernel_height > 0,
        message="kernel size should be greater than zero",
    )
    ivy.assertions.check_true(
        dilation_width > 0 and dilation_height > 0,
        message="dilation should be greater than zero",
    )
    ivy.assertions.check_true(
        pad_width >= 0 and pad_height >= 0, message="padding should be non-negative"
    )
    ivy.assertions.check_true(
        stride_width > 0 and stride_height > 0,
        message="stride should be greater than zero",
    )

    input = ivy.asarray(input)
    ndim = input.ndim

    valid_dims = input.shape[1] != 0 and input.shape[2] != 0
    ivy.assertions.check_true(
        (ndim == 3 and input.shape[0] != 0 and valid_dims)
        or (ndim == 4 and valid_dims and input.shape[3] != 0),
        message="expected 3D or 4D (batch mode) tensor "
        "with possibly 0 batch size "
        "and other non-zero dimensions for input",
    )

    dim_batch = 0
    if ndim == 3:
        dim_batch = -1

    input_height = input.shape[dim_batch + 2]
    input_width = input.shape[dim_batch + 3]
    output_height = int(
        _div_rtn(
            input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1),
            stride_height,
        )
        + 1
    )
    output_width = int(
        _div_rtn(
            input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1),
            stride_width,
        )
        + 1
    )

    ivy.assertions.check_true(
        output_width >= 1 and output_height >= 1,
        message="calculated shape of the array " "of sliding blocks is non-positive",
    )

    batched_input = True
    if input.ndim == 3:
        batched_input = False
        input = ivy.reshape(input, (1, input.shape[0], input.shape[1], input.shape[2]))

    batch_size = input.shape[0]
    n_input_channels = input.shape[1]
    input_height = input.shape[2]
    input_width = input.shape[3]
    output_height = int(
        (input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1))
        / stride_height
        + 1
    )
    output_width = int(
        (input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1))
        / stride_width
        + 1
    )
    n_output_channels = n_input_channels * kernel_width * kernel_height
    output_length = output_height * output_width

    output = ivy.zeros((batch_size, int(n_output_channels), output_length))

    height_col = output_height
    width_col = output_width
    channels_col = int(n_input_channels * kernel_height * kernel_width)

    for elt in range(batch_size):
        data_im = input[elt]
        data_col = output[elt]

        for c_col in range(channels_col):
            w_offset = c_col % kernel_width
            h_offset = int((c_col / kernel_width) % kernel_height)
            c_im = int(c_col / kernel_height / kernel_width)

            for h_col in range(height_col):
                h_im = h_col * stride_height - pad_height + h_offset * dilation_height

                for w_col in range(width_col):
                    w_im = w_col * stride_width - pad_width + w_offset * dilation_width

                    if 0 <= h_im < input_height and 0 <= w_im < input_width:
                        data_col[h_col, c_col + w_col] = data_im[c_im, h_im, w_im]

    if not batched_input:
        output = ivy.squeeze(output, axis=0)

    return output


@to_ivy_arrays_and_back
def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):

    output_size = ivy.repeat(ivy.asarray(output_size), 2)[:2]
    kernel_size = ivy.repeat(ivy.asarray(kernel_size), 2)[:2]
    dilation = ivy.repeat(ivy.asarray(dilation), 2)[:2]
    padding = ivy.repeat(ivy.asarray(padding), 2)[:2]
    stride = ivy.repeat(ivy.asarray(stride), 2)[:2]

    output_height, output_width = output_size
    kernel_height, kernel_width = kernel_size
    dilation_height, dilation_width = dilation
    pad_height, pad_width = padding
    stride_height, stride_width = stride

    ivy.assertions.check_true(
        output_width >= 1 or output_height >= 1,
        message="expected output spatial size to be positive",
    )
    ivy.assertions.check_true(
        kernel_width > 0 and kernel_height > 0,
        message="kernel size should be greater than zero",
    )
    ivy.assertions.check_true(
        stride_width > 0 and stride_height > 0,
        message="stride should be greater than zero",
    )
    ivy.assertions.check_true(
        dilation_width > 0 and dilation_height > 0,
        message="dilation should be greater than zero",
    )

    input = ivy.asarray(input)
    ndim = input.ndim

    ivy.assertions.check_true(
        (ndim == 2 and input.shape[0] and input.shape[1])
        or (ndim == 3 and input.shape[1] and input.shape[2]),
        message="expected 2D or 3D (batch mode) tensor "
        "with possibly 0 batch size and "
        "non-zero dimensions for input",
    )

    dim_batch = 0
    if input.ndim == 3:
        dim_batch = -1

    n_input_channels = input.shape[dim_batch + 1]

    ivy.assertions.check_true(
        n_input_channels % (kernel_width * kernel_height) != 0,
        message="expected size of input's dimension 1 to be "
        "divisible by the product of kernel_size",
    )

    input_length = input.shape[dim_batch + 2]

    blocks_height = int(
        _div_rtn(
            output_height
            + 2 * pad_height
            - (dilation_height * (kernel_height - 1) - 1),
            stride_height,
        )
        + 1
    )
    blocks_width = int(
        _div_rtn(
            output_width + 2 * pad_width - (dilation_width * (kernel_width - 1) - 1),
            stride_width,
        )
        + 1
    )

    ivy.assertions.check_true(
        input_length != (blocks_height * blocks_width),
        message="expected size of input's dimension 2 to "
        "match the calculated number of sliding blocks",
    )

    batched_input = True
    if input.ndim == 2:
        batched_input = False
        input = ivy.reshape(input, (1, input.shape[0], input.shape[1]))

    batch_size = input.shape[0]
    n_output_channels = int(n_input_channels / (kernel_width * kernel_height))

    output = ivy.zeros(
        (batch_size, n_output_channels, int(output_height), int(output_width))
    )

    height_col = int(
        (output_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1))
        / stride_height
        + 1
    )
    width_col = int(
        (output_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1))
        / stride_width
        + 1
    )
    channels_col = int(n_output_channels * kernel_height * kernel_width)

    for elt in range(batch_size):
        data_col = input[elt]
        data_im = output[elt]

        for c_col in range(channels_col):
            w_offset = c_col % kernel_width
            h_offset = int((c_col / kernel_width) % kernel_height)
            c_im = int(c_col / kernel_height / kernel_width)

            for h_col in range(height_col):
                h_im = h_col * stride_height - pad_height + h_offset * dilation_height

                for w_col in range(width_col):
                    w_im = w_col * stride_width - pad_width + w_offset * dilation_width

                    if 0 <= h_im < output_height and 0 <= w_im < output_width:
                        data_im[c_im, h_im, w_im] += data_col[h_col, c_col + w_col]

    if not batched_input:
        output = ivy.squeeze(output, axis=0)

    return output
