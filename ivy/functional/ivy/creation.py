# global
from typing import Union, Tuple, Optional, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def zeros(shape: Union[int, Tuple[int], List[int]],
          dtype: Optional[ivy.Dtype] = None,
          device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape:
       output array shape.
    dtype:
       output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device:
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    out:
       an array containing zeros.

    """
    return _cur_framework().zeros(shape, dtype, device)


def ones(shape: Union[int, Tuple[int], List[int]],
         dtype: Optional[ivy.Dtype] = None,
         device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Returns a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape:
        output array shape.
    dtype:
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device:
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    out:
        an array containing ones.
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


def tril(x: Union[ivy.Array, ivy.NativeArray],
         k: int = 0) \
         -> ivy.Array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) x.

    Parameters
    ----------
    x:
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
    k:
        diagonal above which to zero elements. If k = 0, the diagonal is the main diagonal. If k < 0, the diagonal is
        below the main diagonal. If k > 0, the diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    out:
        an array containing the lower triangular part(s). The returned array must have the same shape and data type as
        x. All elements above the specified diagonal k must be zeroed. The returned array should be allocated on the
        same device as x.
    """
    return _cur_framework(x).tril(x, k)


def triu(x: Union[ivy.Array, ivy.NativeArray],
         k: int = 0) \
         -> ivy.Array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) x.

    Parameters
    ----------
    x:
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
    k:
        diagonal below which to zero elements. If k = 0, the diagonal is the main diagonal. If k < 0, the diagonal is 
        below the main diagonal. If k > 0, the diagonal is above the main diagonal. Default: 0.

    Returns
    -------
    out:
        an array containing the upper triangular part(s). The returned array must have the same shape and data type as 
        x. All elements below the specified diagonal k must be zeroed. The returned array should be allocated on the 
        same device as x.
    """
    return _cur_framework(x).triu(x, k)
    

def empty(shape: Union[int, Tuple[int],List[int]],
          dtype: Optional[ivy.Dtype] = None,
          device: Optional[ivy.Device] = None)\
        -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.
    :param shape: output array shape.
    :param dtype: output array data type. If dtype is None, the output array data type must be the default
                  floating-point data type. Default: None.
    :param device: device on which to place the created array. Default: None.
    :return: an uninitialized array having a specified shape
    """
    return _cur_framework().empty(shape, dtype, device)


# Extra #
# ------#
