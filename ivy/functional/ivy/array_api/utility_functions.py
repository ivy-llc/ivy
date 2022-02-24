# global
from typing import Union, Optional, Tuple, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# noinspection PyShadowingBuiltins
def all(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False)\
        -> ivy.Array:
    """
    Tests whether all input array elements evaluate to True along a specified axis.

    :param x: input array.
    :param axis: axis or axes along which to perform a logical AND reduction. By default, a logical AND reduction must
        be performed over the entire array. If a tuple of integers, logical AND reductions must be performed over multiple
        axes. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of dimensions) of x.
        If an axis is specified as a negative integer, the function must determine the axis along which to perform a
        reduction by counting backward from the last dimension (where -1 refers to the last dimension). If provided an
        invalid axis, the function must raise an exception. Default: None.
    :param  keepdims: If True, the reduced axes (dimensions) must be included in the result as singleton dimensions,
        and, accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False,
        the reduced axes (dimensions) must not be included in the result. Default is False.
    :return: if a logical AND reduction was performed over the entire array, the returned array must be a
            zero-dimensional array containing the test result; otherwise, the returned array must be a
            non-zero-dimensional array containing the test results. The returned array must have a data type of bool.
    """
    return _cur_framework(x).all(x, axis, keepdims)


def any(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False)\
        -> ivy.Array:
    """
        Tests whether any input array element evaluate to True along a specified axis.
        :param x: input array.
        :param axis: axis or axes along which to perform a logical OR reduction.
        By default, a logical OR reduction must be performed over the entire array.
        If a tuple of integers, logical OR reductions must be performed over multiple axes.
        A valid axis must be an integer on the interval [-N, N), where N is the rank (number of dimensions) of x.
        If an axis is specified as a negative integer, the function must determine the axis along which to perform a
        reduction by counting backward from the last dimension (where -1 refers to the last dimension).
        If provided an invalid axis, the function must raise an exception. Default: None.
        :param  keepdims: If True, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included in the result. Default is False.
        :return: if a logical OR reduction was performed over the entire array, the returned array must be a
        zero-dimensional array containing the test result; otherwise, the returned array must be a
        non-zero-dimensional array containing the test results. The returned array must have a data type of bool.
        """
    return _cur_framework(x).any(x, axis, keepdims)
