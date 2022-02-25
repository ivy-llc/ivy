# global
from typing import Union, Tuple, Optional


# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def zeros(shape: Union[int, Tuple[int, ...]],
          dtype: Optional[ivy.Dtype] = None,
          device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.

    :param shape: output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default
                  floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :return: an array containing zeros.
    """
    return _cur_framework().zeros(shape, dtype, device)


def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[ivy.Dtype] = 'float32',
         device: Optional[ivy.Device] = None) \
        -> ivy.Array:
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: Shape of the new array, e.g. (2, 3).
    :type shape: iterable int or int
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
    Default is 'float32'.
    :type dtype: data-type string, optional
    :param device: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc..
    :type device: ivy.Device
    :return: Tensor of ones with the given shape and dtype.
    """
    return _cur_framework().ones(shape, dtype, device)
