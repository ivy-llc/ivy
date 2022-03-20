"""
Collection of dtype Ivy functions.
"""


# global
import importlib
import numpy as np
from typing import Union
from numbers import Number

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

default_dtype_stack = list()
default_float_dtype_stack = list()
default_int_dtype_stack = list()


class DefaultDtype:
    # noinspection PyShadowingNames
    def __init__(self, dtype):
        self._dtype = dtype

    def __enter__(self):
        set_default_dtype(self._dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_dtype()
        return self


class DefaultFloatDtype:
    # noinspection PyShadowingNames
    def __init__(self, float_dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_float_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_float_dtype()
        return self


class DefaultIntDtype:
    # noinspection PyShadowingNames
    def __init__(self, float_dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_int_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        return self


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


# Dtype Format Conversion #
# ------------------------#

def dtype_from_str(dtype_in: Union[ivy.Dtype, str])\
        -> ivy.Dtype:
    """
    Convert data type string representation to native data type.

    :param dtype_in: The data type string to convert to native data type.
    :type dtype_in: str
    :return: data type e.g. ivy.float32.
    """
    return _cur_framework(None).dtype_from_str(dtype_in)


# Default Dtype #
# --------------#

# noinspection PyShadowingNames
def _assert_dtype_correct_formatting(dtype):
    assert 'int' in dtype or 'float' in dtype or 'bool' in dtype


# noinspection PyShadowingNames
def default_dtype(dtype=None, item=None, as_str=False):
    """
    Return the input dtype if provided, otherwise return the global default dtype.
    """
    if ivy.exists(dtype):
        _assert_dtype_correct_formatting(ivy.dtype_to_str(dtype))
        return dtype
    elif ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_float_dtype(item):
            return default_float_dtype(as_str=as_str)
        elif ivy.is_int_dtype(item):
            return default_int_dtype(as_str=as_str)
        elif as_str:
            return 'bool'
        else:
            return dtype_from_str('bool')
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = 'float32'
    else:
        ret = default_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# noinspection PyShadowingNames
def set_default_dtype(dtype):
    dtype = ivy.dtype_to_str(dtype)
    _assert_dtype_correct_formatting(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


def unset_default_dtype():
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


# Default Float Dtype #
# --------------------#

# noinspection PyShadowingNames
def _assert_float_dtype_correct_formatting(dtype):
    assert 'float' in dtype


# noinspection PyShadowingNames
def default_float_dtype(float_dtype=None, as_str=False):
    """
    Return the input float dtype if provided, otherwise return the global default float dtype.
    """
    if ivy.exists(float_dtype):
        _assert_float_dtype_correct_formatting(ivy.dtype_to_str(float_dtype))
        return float_dtype
    global default_float_dtype_stack
    if not default_float_dtype_stack:
        def_dtype = default_dtype()
        if ivy.is_float_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = 'float32'
    else:
        ret = default_float_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# noinspection PyShadowingNames
def set_default_float_dtype(float_dtype):
    float_dtype = ivy.dtype_to_str(float_dtype)
    _assert_float_dtype_correct_formatting(float_dtype)
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


def unset_default_float_dtype():
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


# Default Int Dtype #
# ------------------#

# noinspection PyShadowingNames
def _assert_int_dtype_correct_formatting(dtype):
    assert 'int' in dtype


# noinspection PyShadowingNames
def default_int_dtype(int_dtype=None, as_str=False):
    """
    Return the input int dtype if provided, otherwise return the global default int dtype.
    """
    if ivy.exists(int_dtype):
        _assert_int_dtype_correct_formatting(ivy.dtype_to_str(int_dtype))
        return int_dtype
    global default_int_dtype_stack
    if not default_int_dtype_stack:
        def_dtype = default_dtype()
        if ivy.is_int_dtype(def_dtype):
            ret = def_dtype
        else:
            ret = 'int32'
    else:
        ret = default_int_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# noinspection PyShadowingNames
def set_default_int_dtype(int_dtype):
    int_dtype = ivy.dtype_to_str(int_dtype)
    _assert_int_dtype_correct_formatting(int_dtype)
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


def unset_default_int_dtype():
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)
