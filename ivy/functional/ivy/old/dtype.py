"""
Collection of dtype Ivy functions.
"""


# global
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Casting #
# --------#

# noinspection PyShadowingNames
def cast(x: Union[ivy.Array, ivy.NativeArray], dtype: ivy.Dtype)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Casts an array to a specified type.

    :param x: Input array containing elements to cast.
    :type x: array
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
            If not given, then the type will be determined as the minimum type required to hold the objects in the
            sequence.
    :type dtype: data-type string
    :return: A new array of the same shape as input array a, with data type given by dtype.
    """
    return _cur_framework(x).cast(x, dtype)


astype = cast


# Queries #
# --------#

# noinspection PyShadowingBuiltins
def closest_valid_dtype(type: Union[ivy.Dtype, str, None]):
    """
    Determines the closest valid datatype to the datatype passed as input.

    :param type: The data type for which to check the closest valid type for.
    :return: The closest valid data type as a native ivy.Dtype
    """
    return _cur_framework(type).closest_valid_dtype(type)
