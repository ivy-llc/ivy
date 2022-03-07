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
    Returns a new array having a specified ``shape`` and filled with zeros.
    Parameters
    ----------
    shape: Union[int, Tuple[int, ...]]
       output array shape.
    dtype: Optional[dtype]
       output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
       device on which to place the created array. Default: ``None``.
    Returns
    -------
    out: array
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
    shape: Union[int, Tuple[int, ...]]
        output array shape.
    dtype: Optional[dtype]
        output array data type. If ``dtype`` is ``None``, the output array data type must be the default floating-point data type. Default: ``None``.
    device: Optional[device]
        device on which to place the created array. Default: ``None``.
    Returns
    -------
    out: array
        an array containing ones.
    """
    return _cur_framework().ones(shape, dtype, device)
