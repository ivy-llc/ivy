import ivy


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """

    Parameters (Same as PyTorch Implementation)
    ----------
    input: input tensor of shape (batch_size, d_in, w)
    weight: filters of shape (d_out, d_in/groups, fw)
    bias:  optional bias of shape (d_out). Default: None
    stride: the stride of the convolving kernel. Can be a single number or a one-element tuple (sW,). Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {‘valid’, ‘same’}, single number or a
             one-element tuple (padW,). Default: 0 padding='valid' is the same as no padding.
             padding='same' pads the input so the output has the same shape as the input.
    dilation: the spacing between kernel elements. Can be a single number or a one-element tuple (dW,). Default: 1
    groups: split input into groups, in_channels and out_channels should be divisible by the number of groups. Default: 1

    Returns
    -------
    output tensor of shape (batch_size, d_out, w)
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
        input = ivy.zero_pad(input, pad_width=((0, 0), (0, 0), (padding, padding)))

    out = []
    _weight = ivy.permute_dims(weight, axes=(2, 1, 0))
    _in_chunk = in_channel // groups
    _out_chunk = out_channel // groups
    for i in range(groups):
        _input = ivy.permute_dims(input[:, int(i * _in_chunk):int((i + 1) * _in_chunk), :], axes=(0, 2, 1))
        out.append(
            ivy.conv1d(_input, _weight, stride, padding,
                       dilations=dilation)[:, :, int(i * _out_chunk):int((i + 1) * _out_chunk)]
        )
    out = ivy.concat(out, axis=-1)
    if bias is not None:
        out = ivy.add(out, bias)
    return ivy.permute_dims(out, axes=(0, 2, 1))
