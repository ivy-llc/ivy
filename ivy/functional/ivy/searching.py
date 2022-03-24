# global
from typing import Union, Optional

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#

def argmax(
    x: Union[ivy.Array, ivy.NativeArray], 
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """
    Returns the indices of the maximum values along a specified axis. When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.
    axis:
        axis along which to search. If None, the function must return the index of the maximum value of the flattened array. Default: None.
    keepdims:
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the array.
    out:
        If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.

    Returns
    -------
        if axis is None, a zero-dimensional array containing the index of the first occurrence of the maximum value; otherwise, a non-zero-dimensional array containing the indices of the maximum values. The returned array must have be the default array index data type.
    """
    return _cur_framework(x).argmax(x,axis,keepdims,out)


def argmin(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None
) -> ivy.Array:
    """
    Returns the indices of the minimum values along a specified axis. When the miniumum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x:
        input array. Should have a numeric data type.
    axis:
        axis along which to search. If None, the function must return the index of the minimum value of the flattened array. Default: None.
    keepdims:
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the array.
    out:
        If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.

    Returns
    -------
        if axis is None, a zero-dimensional array containing the index of the first occurrence of the minimum value; otherwise, a non-zero-dimensional array containing the indices of the maximum values. The returned array must have be the default array index data type.
    """
    return _cur_framework(x).argmin(x, axis, keepdims, out)
# Extra #
# ------#
