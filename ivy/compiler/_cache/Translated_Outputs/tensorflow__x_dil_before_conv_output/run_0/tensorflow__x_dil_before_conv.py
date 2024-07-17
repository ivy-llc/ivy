import tensorflow


def tensorflow__x_dil_before_conv(x, dims, x_dilations, data_format):
    x_dilations = [x_dilations] * dims if isinstance(x_dilations, int) else x_dilations
    ag__result_list_0 = []
    for i, x_dil in enumerate(x_dilations):
        if x_dil > 1:
            res = i
            ag__result_list_0.append(res)
    x_dilations_idxs = ag__result_list_0
    if x_dilations_idxs:
        if data_format[-1] == "C":
            offset = 1
        else:
            offset = 2
        for i in x_dilations_idxs:
            h = x.shape[offset + i]
            new_height = h + (h - 1) * (x_dilations[i] - 1)
            h = tensorflow.eye(new_height, dtype=x.dtype)[:: x_dilations[i]]
            x = tensorflow.experimental.numpy.swapaxes(x, offset + i, -1)
            x = tensorflow.matmul(x, h)
            x = tensorflow.experimental.numpy.swapaxes(x, -1, offset + i)
    return x
