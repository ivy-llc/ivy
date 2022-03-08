# global
from typing import Union, Tuple, Any

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

Finfo = None
Iinfo = None


# Dtype Info #
# -----------#

# noinspection PyShadowingBuiltins
def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray])\
        -> Iinfo:
    """
    Machine limits for integer data types.

    :param type: the kind of integer data-type about which to get information.
    :return: iinfo object.
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

    :param type: the kind of floating-point data-type about which to get information.
    :return: finfo object.
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

# AsType #
#--------#

def astype(x: Union[ivy.Array, ivy.NativeArray], dtype: ivy.Dtype, /, *, copy: bool = True)\
    -> Union[ivy.Array, ivy.NativeArray]:

    """
    Copies an array to a specified data type irrespective of Type Promotion Rules.

    :param x: array to cast.
    :dtype: desired data type.
    :copy: (bool) specifies whether to copy an array when the specified dtype matches the data type of the input array x.
                If True, a newly allocated array must always be returned.
                If False and the specified dtype matches the data type of the input array, the input array must be returned; otherwise, a newly allocated must be returned.
                Default: True.
    :return: an array having the specified data type. The returned array must have the same shape as x.
    """
    return _cur_framework(x).astype(x, dtype)

