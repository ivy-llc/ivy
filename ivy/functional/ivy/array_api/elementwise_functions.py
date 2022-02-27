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

def sqrt(x: Union[ivy.Array, ivy.NativeArray])\
        -> ivy.Array:
    """
    Calculates the square root, having domain [0, +infinity] and codomain [0, +infinity], for each element
    x_i of the input array x. After rounding, each result must be indistinguishable from the infinitely
    precise result (as required by IEEE 754).

    :param x: input array. Should have a floating-point data type.
    :return: an array containing the square root of each element in x. The returned array must have a
             floating-point data type determined by Type Promotion Rules.
    """
    return _cur_framework(x).sqrt(x)