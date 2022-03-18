# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


# Dtype Info #

# noinspection PyShadowingBuiltins
def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray])\
        -> Iinfo:
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type:
        the kind of integer data-type about which to get information.

    Returns
    -------
    out:
        a class with that encapsules the following attributes:
        - **bits**: *int*
          number of bits occupied by the type.
        - **max**: *int*
          largest representable number.
        - **min**: *int*
          smallest representable number.
    """
    return _cur_framework(None).iinfo(type)


# noinspection PyShadowingBuiltins
def finfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray])\
        -> Finfo:
    """
    Machine limits for floating-point data types.

    Parameters
    ----------
    type:
        the kind of floating-point data-type about which to get information.

    Returns
    -------
    out:
        an object having the followng attributes:
        - **bits**: *int*
          number of bits occupied by the floating-point data type.
        - **eps**: *float*
          difference between 1.0 and the next smallest representable floating-point number larger than 1.0 according to the IEEE-754 standard.
        - **max**: *float*
          largest representable number.
        - **min**: *float*
          smallest representable number.
        - **smallest_normal**: *float*
          smallest positive floating-point number with full precision.
    """
    return _cur_framework(None).finfo(type)


def dtype(x: Union[ivy.Array, ivy.NativeArray], as_str: bool = False)\
        -> ivy.Dtype:
    """
    Get the data type for input array x.

    :param x: Tensor for which to get the data type.
    :type x: array
    :param as_str: Whether or not to return the dtype in string format. Default is False.
    :type as_str: bool, optional
    :return: Data type of the array
    """
    return _cur_framework(x).dtype(x, as_str)


# Extra #
# ------#
