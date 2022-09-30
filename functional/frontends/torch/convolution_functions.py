import ivy


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Parameters
    ----------
    input: array
        input array of shape (batch_size, d_in, h, w)
    weight: array
        filters of shape (d_out, d_in/groups, fh, fw)
    bias: array or None
        optional bias of shape (d_out). Default: None
    stride: int or tuple
        the stride of the convolving kernel. Can be a single number or a one-element
        tuple (sH, sW). Default: 1
    padding: int or tuple
        implicit paddings on both sides of the input. Can be a
        string {‘valid’, ‘same’}, single number or a one-element tuple (padH,
        padW). Default: 0 padding='valid' is the same as no padding. padding='same'
        pads the input so the output has the same shape as the input.
    dilation: int
        the spacing between kernel elements. Can be a single number or a one-element
        tuple (dH, dW). Default: 1
    groups: int
        split input into groups, in_channels and out_channels should be divisible by
         the number of groups. Default: 1

    Returns
    -------
    out: array
        output tensor of shape (batch_size, d_out, h_out, w_out)
    """
    in_channel = input.shape[1]
    if in_channel % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    out_channel = weight.shape[0]
    if out_channel % groups != 0:
        raise ValueError("out_channels must be divisible by groups")

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
