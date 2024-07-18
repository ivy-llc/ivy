import tensorflow


def tensorflow__pad_before_conv(x, padding, dims, data_format):
    if isinstance(padding, str):
        return x, padding
    elif isinstance(padding, int):
        pad_list = [(padding, padding)] * dims
    else:
        pad_list = padding
    if data_format[-1] == "C":
        pad_list = [(0, 0), *pad_list, (0, 0)]
    else:
        pad_list = [(0, 0), (0, 0), *pad_list]
    return tensorflow.pad(x, pad_list, "CONSTANT"), "VALID"
