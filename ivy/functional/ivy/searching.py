# global
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.backend_handler import current_backend as _cur_backend
from ivy.func_wrapper import to_native_arrays_and_back, handle_out_argument


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Returns the indices of the maximum values along a specified axis. When the
    maximum value occurs multiple times, only the indices corresponding to the first
    occurrence are returned.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        maximum value of the flattened array. Default  None.
    keepdims
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the array.
    out
        If provided, the result will be inserted into this array. It should be of the
        appropriate shape and dtype.

    Returns
    -------
    ret
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the maximum value; otherwise, a non-zero-dimensional array
        containing the indices of the maximum values. The returned array must have be
        the default array index data type.

    """
    return _cur_backend(x).argmax(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
def argmin(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    
) -> ivy.Array:
    """Returns the indices of the minimum values along a specified axis. When the
    minimum value occurs multiple times, only the indices corresponding to the first
    occurrence are returned.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        minimum value of the flattened array. Default = None.
    keepdims
        if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default = False.
    out
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the minimum value; otherwise, a non-zero-dimensional array
        containing the indices of the minimum values. The returned array must have the
        default array index data type.

    Returns
    -------
    ret
        Array containing the indices of the minimum values across the specified axis.

    Examples
    --------
    
    With :code:`ivy.Array` input:
        
    >>> x = ivy.array([-0., 1., -1.])
    >>> y = ivy.argmin(x)
    >>> print(y)
    tensor([2])
    

    >>> x=ivy.array([[0., 1., -1.],
                     [-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis= 1)
    >>> print(y)
    ivy.array([2, 0])

    >>> x=ivy.array([[0., 1., -1.], 
                     [-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis= 1, keepdims= True)
    >>> print(y)
    ivy.array([[2],
              [0]])

    >>> x=ivy.array([[0., 1., -1.], 
                     [-2., 1., 2.],
                     [1., -2., 0.]])
    >>> y= ivy.zeros((1,3), dtype=ivy.int64)
    >>> ivy.argmin(x, axis= 1, keepdims= True, out= y)
    >>> print(y)
    ivy.array([[2],
               [0],
               [1]])


    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-0., 1., -1.])
    >>> y = ivy.argmin(x)
    >>> print(y)
    ivy.array([2])

    """
    
    
    
    return _cur_backend(x).argmin(x, axis, keepdims, out=out)


@to_native_arrays_and_back
def nonzero(x: Union[ivy.Array, ivy.NativeArray]) -> Tuple[ivy.Array]:
    """Returns the indices of the array elements which are non-zero.

    Parameters
    ----------
    x
        input array. Must have a positive rank. If `x` is zero-dimensional, the function
        must raise an exception.

    Returns
    -------
    ret
        a tuple of `k` arrays, one for each dimension of `x` and each of size `n`
        (where `n` is the total number of non-zero elements), containing the indices of
        the non-zero elements in that dimension. The indices must be returned in
        row-major, C-style order. The returned array must have the default array index
        data type.

    """
    return _cur_backend(x).nonzero(x)


@to_native_arrays_and_back
@handle_out_argument
def where(
    condition: Union[ivy.Array, ivy.NativeArray],
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Returns elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition
        Where True, yield x1, otherwise yield x2.
    x1
        values from which to choose when condition is True.
    x2
        values from which to choose when condition is False.

    Returns
    -------
    ret
        An array with elements from x1 where condition is True, and elements from x2
        elsewhere.

    """
    return _cur_backend(x1).where(condition, x1, x2)


# Extra #
# ------#
