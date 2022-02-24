# local
import ivy
from typing import Union
from ivy.framework_handler import current_framework as _cur_framework


def isfinite(x: Union[ivy.Array, ivy.NativeArray]) -> ivy.Array:
    """
    Tests each element x_i of the input array x to determine if finite (i.e., not NaN and not equal to positive
    or negative infinity).

    :param x: input array. Should have a numeric data type.
    :return: an array containing test results. An element out_i is True if x_i is finite and False otherwise.
             The returned array must have a data type of bool.
    """
    return _cur_framework(x).isfinite(x)


def isnan(x: Union[ivy.Array, ivy.NativeArray])\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns boolean map at locations where the input is not a number (nan).

    :param x: Input array.
    :type x: array
    :return: Boolean values for where the values of the array are nan.
    """
    return _cur_framework(x).isnan(x)
