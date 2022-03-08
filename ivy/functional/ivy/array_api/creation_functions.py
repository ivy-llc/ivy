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


def tril(x: Union[ivy.Array, ivy.NativeArray],
         k: int = 0) \
         -> ivy.Array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) x.

    :param x: input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
    :param k: diagonal above which to zero elements. If k = 0, the diagonal is the main diagonal. If k < 0,
              the diagonal is below the main diagonal. If k > 0, the diagonal is above the main diagonal. Default: 0.
    :return: an array containing the lower triangular part(s). The returned array must have the same shape and
             data type as x. All elements above the specified diagonal k must be zeroed.
             The returned array should be allocated on the same device as x.
    """
    return _cur_framework(x).tril(x, k)
