from .tensorflow__helpers import tensorflow_conv_general_dilated


def tensorflow__conv(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    dims = len(input.shape) - 2
    if isinstance(padding, str):
        padding = padding.upper()
    elif isinstance(padding, int):
        padding = [*[(padding, padding) for _ in range(dims)]]
    else:
        padding = [*[(p, p) for p in padding]]
    ret = tensorflow_conv_general_dilated(
        input,
        weight,
        stride,
        padding,
        dims=dims,
        data_format="channel_last",
        filter_format="channel_last",
        dilations=dilation,
        feature_group_count=groups,
        bias=bias,
    )
    return ret
