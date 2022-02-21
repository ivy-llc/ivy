# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def isfinite(x: ivy.Array) -> ivy.Array:
    """
    Tests each element x_i of the input array x to determine if finite (i.e., not NaN and not equal to positive
    or negative infinity).

    :param x: input array. Should have a numeric data type.
    :return: an array containing test results. An element out_i is True if x_i is finite and False otherwise.
             The returned array must have a data type of bool.
    """
    return _cur_framework(x).isfinite(x)
