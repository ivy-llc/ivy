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


def less(x1: Union[ivy.Array, ivy.NativeArray],x2: Union[ivy.Array, ivy.NativeArray], f: ivy.Framework = None)\
        -> ivy.Array:
    """
    Computes the truth value of x1_i < x2_i for each element x1_i of the input array x1 with the respective 
    element x2_i of the input array x2.

    :param x1: Input array.
    :type x1: array
    :param x2: Input array.
    :type x2: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: an array containing the element-wise results. The returned array must have a data type of bool.
    """
    return _cur_framework(x1,f=f).less(x1,x2)   