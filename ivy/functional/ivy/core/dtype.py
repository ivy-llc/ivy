"""
Collection of dtype Ivy functions.
"""


# global
import abc
import importlib
from typing import Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

Finfo = None
Iinfo = None


# noinspection PyShadowingNames
def cast(x: Union[ivy.Array, ivy.NativeArray], dtype: ivy.Dtype, f: ivy.Framework = None)\
        -> Union[ivy.Array, ivy.NativeArray]:
    """
    Casts an array to a specified type.

    :param x: Input array containing elements to cast.
    :type x: array
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
            If not given, then the type will be determined as the minimum type required to hold the objects in the
            sequence.
    :type dtype: data-type string
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: A new array of the same shape as input array a, with data type given by dtype.
    """
    return _cur_framework(x, f=f).cast(x, dtype)


def dtype(x: Union[ivy.Array, ivy.NativeArray], as_str: bool = False, f: ivy.Framework = None)\
        -> ivy.Dtype:
    """
    Get the data type for input array x.

    :param x: Tensor for which to get the data type.
    :type x: array
    :param as_str: Whether or not to return the dtype in string format. Default is False.
    :type as_str: bool, optional
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Data type of the array
    """
    return _cur_framework(x, f=f).dtype(x, as_str)


def is_int_dtype(dtype_in: Union[ivy.Dtype, str]):
    """
    Determine whether the input data type is an int data-type.

    :param dtype_in: Datatype to test
    :return: Whether or not the data type is an integer data type
    """
    return 'int' in dtype_to_str(dtype_in)


def is_float_dtype(dtype_in: Union[ivy.Dtype, str]):
    """
    Determine whether the input data type is an float data-type.

    :param dtype_in: Datatype to test
    :return: Whether or not the data type is a floating point data type
    """
    return 'float' in dtype_to_str(dtype_in)


def valid_dtype(dtype_in: Union[ivy.Dtype, str]):
    """
    Determines whether the provided data type is support by the current framework.

    :param dtype_in: The data type for which to check for backend support
    :return: Boolean, whether or not the data-type string is supported.
    """
    return ivy.dtype_to_str(dtype_in) in ivy.valid_dtype_strs


def invalid_dtype(dtype_in: Union[ivy.Dtype, str]):
    """
    Determines whether the provided data type is not support by the current framework.

    :param dtype_in: The data type for which to check for backend non-support
    :return: Boolean, whether the data-type string is un-supported.
    """
    return ivy.dtype_to_str(dtype_in) in ivy.invalid_dtype_strs


def convert_dtype(dtype_in: Union[ivy.Dtype, str], backend: str):
    """
    Converts a data type from one backend framework representation to another.

    :param dtype_in: The data-type to convert, in the specified backend representation
    :type dtype_in: data type
    :param backend: The backend framework the dtype_in is represented in.
    :type backend: str
    :return: The data-type in the current ivy backend format
    """
    valid_backends = ['numpy', 'jax', 'tensorflow', 'torch', 'mxnet']
    if backend not in valid_backends:
        raise Exception('Invalid backend passed, must be one of {}'.format(valid_backends))
    ivy_backend = importlib.import_module('ivy.functional.backends.{}'.format(backend))
    return ivy.dtype_from_str(ivy_backend.dtype_to_str(dtype_in))


def dtype_to_str(dtype_in: Union[ivy.Dtype, str], f: ivy.Framework = None)\
        -> str:
    """
    Convert native data type to string representation.

    :param dtype_in: The data type to convert to string.
    :type dtype_in: data type
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: data type string 'float32'
    """
    return _cur_framework(None, f=f).dtype_to_str(dtype_in)


def dtype_from_str(dtype_in: Union[ivy.Dtype, str], f: ivy.Framework = None)\
        -> ivy.Dtype:
    """
    Convert data type string representation to native data type.

    :param dtype_in: The data type string to convert to native data type.
    :type dtype_in: str
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: data type e.g. ivy.float32.
    """
    return _cur_framework(None, f=f).dtype_from_str(dtype_in)


# noinspection PyShadowingBuiltins
def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray], f: ivy.Framework = None) -> ivy.Iinfo:
    """
    Machine limits for integer data types.

    :param type: the kind of integer data-type about which to get information.
    :param f: Machine learning framework. Inferred from inputs if None.
    :return: out – object with the machine limits for integer data types.
    """
    return _cur_framework(None, f=f).iinfo(type)


# noinspection PyShadowingBuiltins
def finfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray], f: ivy.Framework = None) -> ivy.Finfo:
    """
    Machine limits for floating-point data types.

    :param type: the kind of floating-point data-type about which to get information.
    :param f: Machine learning framework. Inferred from inputs if None.
    :return: out – object with the machine limits for floating-point data types.
    """
    return _cur_framework(None, f=f).finfo(type)
