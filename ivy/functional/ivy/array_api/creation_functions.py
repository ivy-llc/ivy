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

def ones_like( x: Union[ivy.Array, ivy.NativeArray],
              dtype: Optional[Union[ivy.Dtype, str]] = None,
              dev: Optional[Union[ivy.Device, str]] = None,
              ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns a new array filled with ones and having the same shape as an input array x.

    :param x: Input array from which to derive the output array shape.
    :param dtype: Output array data type. If dtype is None, the output array data type must be inferred from x.
    Default: None.
    :param dev: device on which to place the created array. If device is None, the output array device must be inferred from x.
    Default: None.
    :return: An array having the same shape as x and filled with ones.
    """
    return _cur_framework(x).ones_like(x, dtype, dev)

