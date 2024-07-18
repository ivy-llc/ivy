from numpy.core.numeric import normalize_axis_tuple


def tensorflow__calculate_out_shape(axis, array_shape):
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_dims = len(axis) + len(array_shape)
    norm_axis = normalize_axis_tuple(axis, out_dims)
    shape_iter = iter(array_shape)
    ag__result_list_0 = []
    for current_ax in range(out_dims):
        res = 1 if current_ax in norm_axis else next(shape_iter)
        ag__result_list_0.append(res)
    out_shape = ag__result_list_0
    return out_shape
