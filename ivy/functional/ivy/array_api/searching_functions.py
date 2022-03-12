import ivy
from typing import Union, Optional
from ivy.framework_handler import current_framework as _cur_framework


def argmax(
    x: Union[ivy.Array, ivy.NativeArray], 
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Returns the indices of the maximum values along an axis.
    :param x: input array.
    :param axis(Optional): By default, the index is into the flattened array, otherwise along the specified axis.
    :param out(Optional): If provided, the result will be inserted into this array. It should be of the appropriate shape and dtype.
    :param keepdims(Optional):If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the array.
    :return: Array of indices into the array. It has the same shape as x.shape with the dimension along axis removed. If keepdims is set to True, then the size of axis will be 1 with the resulting array having same shape as x.shape.
    """
    return _cur_framework(x).argmax(x)