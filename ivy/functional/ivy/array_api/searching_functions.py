# global
from typing import Optional, Union

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework 
#connection test
def argmin(x: Union[ivy.Array, ivy.NativeArray],
            axis: Optional[int] = None, 
            out: Optional[ivy.Array] = None, 
            keepdims: Optional[bool] = False) \
        -> ivy.Array:
    """
    Returns the indices of the minimum values along a specified axis. When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned

    :param x: input array. Should have a numeric data type.
    :param axis: axis along which to search. If None, the function must return the index of the minimum value of the flattened 
                array. Default = None.
    :param keepdims: if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, 
                accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False, 
                the reduced axes (dimensions) must not be included in the result. Default = False.
    :param out: if axis is None, a zero-dimensional array containing the index of the first occurrence of the minimum value; 
                otherwise, a non-zero-dimensional array containing the indices of the minimum values. 
                The returned array must have the default array index data type.
    :return: Array containing the indices of the minimum values across the specified axis.
    """

    return _cur_framework(x).argmin(x, axis, out, keepdims)
