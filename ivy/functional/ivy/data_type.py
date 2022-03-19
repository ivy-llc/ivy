# global
import numpy as np
from typing import Union
from numbers import Number

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

def dtype_bits(dtype_in: Union[ivy.Dtype, str]) -> int:
    """
    Get the number of bits used for representing the input data type.

    :param dtype_in: The data type to determine the number of bits for.
    :return: The number of bits used to represent the data type.
    """
    return _cur_framework(dtype_in).dtype_bits(dtype_in)


def dtype_to_str(dtype_in: Union[ivy.Dtype, str])\
        -> str:
    """
    Convert native data type to string representation.

    :param dtype_in: The data type to convert to string.
    :type dtype_in: data type
    :return: data type string 'float32'
    """
    return _cur_framework(None).dtype_to_str(dtype_in)


def is_int_dtype(dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number])\
        -> bool:
    """
    Determine whether the input data type is an int data-type.

    :param dtype_in: Datatype to test
    :return: Whether or not the data type is an integer data type
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return 'int' in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return True if isinstance(dtype_in, (int, np.integer)) and not isinstance(dtype_in, bool) else False
    elif isinstance(dtype_in, (list, tuple, dict)):
        return True if ivy.nested_indices_where(dtype_in, lambda x: isinstance(x, (int, np.integer))) else False
    return 'int' in dtype_to_str(dtype_in)


def is_float_dtype(dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number])\
        -> bool:
    """
    Determine whether the input data type is an float data-type.

    :param dtype_in: Datatype to test
    :return: Whether or not the data type is a floating point data type
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return 'float' in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return True if isinstance(dtype_in, (float, np.floating)) else False
    elif isinstance(dtype_in, (list, tuple, dict)):
        return True if ivy.nested_indices_where(dtype_in, lambda x: isinstance(x, (float, np.floating))) else False
    return 'float' in dtype_to_str(dtype_in)


def valid_dtype(dtype_in: Union[ivy.Dtype, str, None])\
        -> bool:
    """
    Determines whether the provided data type is support by the current framework.

    :param dtype_in: The data type for which to check for backend support
    :return: Boolean, whether or not the data-type string is supported.
    """
    if dtype_in is None:
        return True
    return ivy.dtype_to_str(dtype_in) in ivy.valid_dtype_strs


def invalid_dtype(dtype_in: Union[ivy.Dtype, str, None])\
        -> bool:
    """
    Determines whether the provided data type is not support by the current framework.

    :param dtype_in: The data type for which to check for backend non-support
    :return: Boolean, whether the data-type string is un-supported.
    """
    if dtype_in is None:
        return False
    return ivy.dtype_to_str(dtype_in) in ivy.invalid_dtype_strs
