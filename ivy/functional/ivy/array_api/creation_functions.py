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


def zeros_like(x: Union[ivy.Array, ivy.NativeArray],
               dtype: ivy.Dtype = None,
               dev: ivy.Device = None,
               ) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Returns an array of zeros with the same shape and type as x, unless dtype provided which overrides.

    :param x: The shape and data-type of x define these same attributes of the returned array.
    :type x: array
    :param dtype: The desired data-type for the array in string format, i.e. 'float32' or 'int64'.
                    If not given, then the type of the original array is used.
    :type dtype: data-type string, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: ivy.Device, optional
    :return: Tensor of zeros with the same shape and type as a, unless dtype provided which overrides.
    """
    return _cur_framework(x).zeros_like(x, dtype, dev)



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
