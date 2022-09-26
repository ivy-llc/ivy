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
