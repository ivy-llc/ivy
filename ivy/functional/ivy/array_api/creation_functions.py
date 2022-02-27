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


def linspace(start: Union[int, float],
             stop: Union[int, float],
             num: int,
             dtype: Optional[ivy.Dtype] = None,
             device: Optional[ivy.Device] = None,
             endpoint: bool = True) \
             -> ivy.Array:
    """
    Returns evenly spaced numbers over a specified interval.

    :param start: the start of the interval.
    :param stop: the end of the interval. If endpoint is False, the function must generate a
                 sequence of num+1 evenly spaced numbers starting with start and ending with stop and exclude the
                 stop from the returned array such that the returned array consists of evenly spaced numbers over the
                 half-open interval [start, stop). If endpoint is True, the output array must consist of
                 evenly spaced numbers over the closed interval [start, stop]. Default: True.
    :param num: number of samples. Must be a non-negative integer value;
                otherwise, the function must raise an exception.
    :param dtype: output array data type. If dtype is None, the output array data type must be the
                  default floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :param endpoint: boolean indicating whether to include stop in the interval. Default: True.
    :return: a one-dimensional array containing evenly spaced values.
    """
    return _cur_framework().linspace(start, stop, num, dtype, device, endpoint)
