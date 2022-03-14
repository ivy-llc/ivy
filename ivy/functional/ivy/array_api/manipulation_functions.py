# global
from typing import Union, Optional, Tuple, List

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


def flip(x: Union[ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, Tuple[int], List[int]]] = None)\
        -> ivy.Array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    Parameters
    ----------
    x:
        input array.
    axis:
        axis (or axes) along which to flip. If ``axis`` is ``None``, the function must flip all input array axes. If ``axis`` is negative, the function must count from the last dimension. If provided more than one axis, the function must flip only the specified axes. Default: ``None``.

    Returns
    -------
    out:
        an output array having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
    """

    return _cur_framework(x).flip(x, axis)

def stack(arrays: Union[ivy.Array, ivy.NativeArray, Tuple[ivy.Array], Tuple[ivy.NativeArray], List[ivy.Array], List[ivy.NativeArray]],
          axis: Optional[int] = None) \
          -> ivy.Array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays
        input arrays to join. Each array must have the same shape.
    axis:
        axis along which the arrays will be joined. Providing an ``axis`` specifies the index of the new axis in the dimensions of the result. For example, if ``axis`` is ``0``, the new axis will be the first dimension and the output array will have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the second dimension and the output array will have shape ``(A, N, B, C)``; and, if ``axis`` is ``-1``, the new axis will be the last dimension and the output array will have shape ``(A, B, C, N)``. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.

    Returns
    --------
    out: array
        an output array having rank ``N+1``, where ``N`` is the rank (number of dimensions) of ``x``. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.
        .. note::
           This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
    """

    return _cur_framework(arrays).stack(arrays, axis)
