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


def eye(n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[ivy.Dtype] = None,
        device: Optional[ivy.Device] = None) \
        -> ivy.Array:
    """
    Returns a two-dimensional array with ones on the k h diagonal and zeros elsewhere.

    Parameters
    :param n_rows: number of rows in the output array.
    :param n_cols: number of columns in the output array. If None, the default number of columns in the output array is
                   equal to n_rows. Default: None.
    :param k: index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal,
              and 0 to the main diagonal. Default: 0.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default floating-
                  point data type. Default: None.
    :return: device on which to place the created array. Default: None.
    :return: an array where all elements are equal to zero, except for the k h diagonal, whose values are equal to one.
    """
    return _cur_framework().eye(n_rows, n_cols, k, dtype, device)