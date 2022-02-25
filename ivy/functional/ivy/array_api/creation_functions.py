# global
from typing import Union, Tuple, Optional, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def zeros(shape: Union[int, Tuple[int], List[int]],
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


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[ivy.Dtype] = None,
         device: Optional[ivy.Device] = None) \
        -> ivy.Array:
    """
    Returns a new array of given shape and type, filled with ones.

    :param shape: output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default
                  floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :return: an array containing ones.
    """
    return _cur_framework().ones(shape, dtype, device)
