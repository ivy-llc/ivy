import ivy
from ivy.framework_handler import current_framework as _cur_framework

def sinh(x: ivy.Array, /) -> ivy.Array:
    """
        Calculates an implementation-dependent approximation to the hyperbolic sine, having domain
        [-infinity, +infinity] and codomain [-infinity, +infinity], for each element x_i
        of the input array x.

        :param x: input array.
        :return: an array containing the hyperbolic sine of each element in x.
        """
    if x.__contains__(float('NaN')):
        return float('NaN')

    if x.__contains__(+0.0):
        return +0.0

    if x.__contains__(-0.0):
        return -0.0

    if x.__contains__(float('inf')):
        return float('inf')

    if x.__contains__(float('-inf')):
        return float('-inf')

    return _cur_framework(x).sinh(x)
