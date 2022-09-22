import ivy


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    in_channel = input.shape[1]
    ivy.assertions.check_equal(
        in_channel % groups, 0, message="in_channels must be divisible by groups"
    )
    out_channel = weight.shape[0]
    ivy.assertions.check_equal(
        out_channel % groups, 0, message="out_channel must be divisible by groups"
    )
    if type(padding) == str:
        padding = padding.upper()
    else:
        if type(padding) == int:
            input = ivy.zero_pad(
                input,
                pad_width=[(0, 0), (0, 0), (padding, padding), (padding, padding)],
            )
        else:
            w_pad, h_pad = padding
            input = ivy.zero_pad(
                input, pad_width=[(0, 0), (0, 0), (w_pad, w_pad), (h_pad, h_pad)]
            )
        padding = "VALID"

    out = []
    _weight = ivy.permute_dims(weight, axes=(2, 3, 1, 0))
    _in_chunk = in_channel // groups
    _out_chunk = out_channel // groups
    for i in range(groups):
        _input = ivy.permute_dims(
            input[:, int(i * _in_chunk) : int((i + 1) * _in_chunk), :],
            axes=(0, 2, 3, 1),
        )
        out.append(
            ivy.conv2d(_input, _weight, stride, padding, dilations=dilation)[
                :, :, :, int(i * _out_chunk) : int((i + 1) * _out_chunk)
            ]
        )
    out = ivy.concat(out, axis=-1)
    if bias is not None:
        out = ivy.add(out, bias)
    return ivy.permute_dims(out, axes=(0, 3, 1, 2))


def _expanding_array(x, length=2):
    if type(x) == int:
        x = x * length
    elif type(x) == tuple:
        if len(x) == 1:
            x = [x] * length
    return x


def _div_rtn(x, y):
    q = x / y
    r = x % y
    if (r != 0) and ((r < 0) != (y < 0)):
        q = q - 1
    return q


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):

    kernel_size = _expanding_array(kernel_size)
    dilation = _expanding_array(dilation)
    padding = _expanding_array(padding)
    stride = _expanding_array(stride)

    kernel_height = int(kernel_size[0])
    kernel_width = int(kernel_size[1])
    dilation_height = int(dilation[0])
    dilation_width = int(dilation[1])
    pad_height = int(padding[0])
    pad_width = int(padding[1])
    stride_height = int(stride[0])
    stride_width = int(stride[1])

    ivy.assertions.check_true(kernel_width > 0 and kernel_height > 0,
                              message="kernel size should be greater than zero")
    ivy.assertions.check_true(dilation_width > 0 and dilation_height > 0,
                              message="dilation should be greater than zero")
    ivy.assertions.check_true(pad_width > 0 and pad_height > 0,
                              message="padding should be greater than zero")
    ivy.assertions.check_true(stride_width > 0 and stride_height > 0,
                              message="stride should be greater than zero")

    ndim = input.dim()

    valid_dims = input.shape[1] != 0 and input.shape[2] != 0
    ivy.assertions.check_true((ndim == 3 and input.shape[0] != 0 and valid_dims) or
                              (ndim == 4 and valid_dims and input.shape[3] != 0),
                              message="Expected 3D or 4D (batch mode) tensor "
                              "with possibly 0 batch size and other non-zero dimensions for input")

    dim_batch = 0
    if ndim == 3:
        dim_batch = -1

    input_height = input.shape[dim_batch + 2]
    input_width = input.shape[dim_batch + 3]
    output_height = _div_rtn(input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1),
                             stride_height) + 1
    output_width = _div_rtn(input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1),
                            stride_width) + 1

    ivy.assertions.check_true(output_width < 1 and output_height < 1,
                              message="calculated shape of the array of sliding blocks is non-positive")

    batched_input = True
    if input.dim() == 3:
        batched_input = False
        input = ivy.reshape(input, {1, input.shape[0], input.shape[1], input.shape[2]})

    batch_size = input.shape[0]
    n_input_channels = input.shape[1]
    input_height = input.shape[2]
    input_width = input.shape[3]
    output_height = int((input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1))
                        / stride_height + 1)
    output_width = int((input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1))
                       / stride_width + 1)
    n_output_channels = n_input_channels * kernel_width * kernel_height
    output_length = output_height * output_width

    output = ivy.zeros({batch_size, n_output_channels, output_length})

    height_col = output_height
    width_col = output_width
    channels_col = n_input_channels * kernel_height * kernel_width

    for elt in range(batch_size):
        data_im = input[elt]
        data_col = output[elt]

        for c_col in range(channels_col):
            w_offset = c_col % kernel_width
            h_offset = (c_col / kernel_width) % kernel_height
            c_im = int(c_col / kernel_height / kernel_width)

            for h_col in range(height_col):
                h_im = h_col * stride_height - pad_height + h_offset * dilation_height

                for w_col in range(width_col):
                    w_im = w_col * stride_width - pad_width + w_offset * dilation_width
                    if (h_im >= 0 and w_im >= 0) and (h_im < input_height and w_im < input_width):
                        data_col[(c_col * height_col + h_col) * width_col + w_col] = \
                            data_im[(c_im * input_height + h_im) * input_width + w_im]

    if not batched_input:
        output = ivy.squeeze(output, axis=0)

    return output
