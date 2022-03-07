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

def arange(start: Union[int, float], /, stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, *,
           dtype: Optional[ivy.Dtype] = None,
           device: Optional[ivy.Device] = None)\
               -> ivy.Array:
    """
    Return evenly spaced values within the half-open interval [start, stop) as a one-dimensional array.
    
    :param start: if stop is specified, the start of interval (inclusive); 
                otherwise, the end of the interval (exclusive). 
                If stop is not specified, the default starting value is 0.
    :param stop: the end of the interval. Default: None.
    :param step: the distance between two adjacent elements (out[i+1] - out[i]). 
                Must not be 0; may be negative, this results in an empty array if stop >= start. Default: 1.
    :param dtype: output array data type. If dtype is None, the output array data type must be inferred from start, 
                stop and step. If those are all integers, the output array dtype must be the default integer dtype; 
                if one or more have type float, then the output array dtype must be the default floating-point data type. 
                Default: None.
    :param device: device on which to place the created array. Default: None.
    
    :return: a one-dimensional array containing evenly spaced values. 
            The length of the output array must be ceil((stop-start)/step) 
            if stop - start and step have the same sign, and length 0 otherwise.
    """
    return _cur_framework().arange(start, stop, step, dtype, device)