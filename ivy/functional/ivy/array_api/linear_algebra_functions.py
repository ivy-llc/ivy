






def cross(x1: Union[ivy.Array, ivy.NativeArray], x2: Union[ivy.Array, ivy.NativeArray], /, *, axis: int = -1) -> ivy.Array:
    """
    Compute and return the cross product of 3-element vectors, it must have the same shape as b
    :param axis: the axis (dimension) of a and b containing the vector for which to compute the cross
    product default is -1
    :type  axis: int
    :param x1: first input, should have a numeric data type
    :type x1: array 
    :param x2: second input, should have a numeric data type 
    :type x2: array
    :return: an array that contains the cross products
    """
    return _cur_framework(a).cross(a, b, axis)
