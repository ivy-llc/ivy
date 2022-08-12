# global
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, int]:
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

    Functional Examples
    --------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([-0., 1., -1.])
    >>> y = ivy.argmax(x)
    >>> print(y)
    ivy.array([1])

    >>> x = ivy.array([-0., 1., -1.])
    >>> ivy.argmax(x,out=x)
    >>> print(x)
    ivy.array([1])

    >>> x=ivy.array([[1., -0., -1.], \
                     [-2., 3., 2.]])
    >>> y = ivy.argmax(x, axis= 1)
    >>> print(y)
    ivy.array([0, 1])

    >>> x=ivy.array([[4., 0., -1.], \
                     [2., -3., 6]])
    >>> y = ivy.argmax(x, axis= 1, keepdims= True)
    >>> print(y)
    ivy.array([[0], \
              [2]])

    >>> x=ivy.array([[4., 0., -1.], \
                     [2., -3., 6], \
                     [2., -3., 6]])
    >>> z= ivy.zeros((1,3), dtype=ivy.int64)
    >>> y = ivy.argmax(x, axis= 1, keepdims= True, out= z)
    >>> print(z)
    ivy.array([[0], \
               [2], \
               [2]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-0., 1., -1.])
    >>> y = ivy.argmax(x)
    >>> print(y)
    ivy.array([1])

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.argmax()
    >>> print(y)
    ivy.array(2)

    """
    return current_backend(x).argmax(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def argmin(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, int]:
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

    Functional Examples
    --------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., -1.])
    >>> y = ivy.argmin(x)
    >>> print(y)
    ivy.array(2)


    >>> x=ivy.array([[0., 1., -1.],[-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis= 1)
    >>> print(y)
    ivy.array([2, 0])

    >>> x=ivy.array([[0., 1., -1.],[-2., 1., 2.]])
    >>> y = ivy.argmin(x, axis= 1, keepdims= True)
    >>> print(y)
    ivy.array([[2],
              [0]])

    >>> x=ivy.array([[0., 1., -1.],[-2., 1., 2.],[1., -2., 0.]])
    >>> y= ivy.zeros((1,3), dtype=ivy.int64)
    >>> ivy.argmin(x, axis= 1, keepdims= True, out= y)
    >>> print(y)
    ivy.array([[2],
               [0],
               [1]])


    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., -1.])
    >>> y = ivy.argmin(x)
    >>> print(y)
    ivy.array(2)


    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., -1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = ivy.argmin(x)
    >>> print(y)
    {a:ivy.array(1),b:ivy.array(0)}


    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., -1.])
    >>> y = x.argmin()
    >>> print(y)
    ivy.array(2)

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0., -1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.argmin()
    >>> print(y)
    {a:ivy.array(1),b:ivy.array(0)}
    """
    return current_backend(x).argmin(x, axis, keepdims, out=out)


@to_native_arrays_and_back
@handle_nestable
def nonzero(x: Union[ivy.Array, ivy.NativeArray], /) -> Tuple[ivy.Array]:
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

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0, 10, 15, 20, -50, 0])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([1, 2, 3, 4]),)

    >>> x = ivy.array([[1, 2], [-1, -2]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([0, 0, 1, 1]), ivy.array([0, 1, 0, 1]))

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[10, 20], [10, 0], [0, 0]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([0, 0, 1]), ivy.array([0, 1, 0]))

    >>> x = ivy.native_array([[0], [1], [1], [0], [1]])
    >>> y = ivy.nonzero(x)
    >>> print(y)
    (ivy.array([1, 2, 4]), ivy.array([0, 0, 0]))

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0,1,2,3,0]), b=ivy.array([[1,1], [0,0]]))
    >>> y = ivy.nonzero(x)
    >>> print(y)
    {
        a: (list[1], <class ivy.array.array.Array> shape=[3]),
        b: (list[2], <class ivy.array.array.Array> shape=[2])
    }

    >>> print(y.a)
    (ivy.array([1, 2, 3]),)

    >>> print(y.b)
    (ivy.array([0, 0]), ivy.array([0, 1]))

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0,0,0,1,1,1])
    >>> y = x.nonzero()
    >>> print(y)
    (ivy.array([3, 4, 5]),)

    Using :code:`ivy.NativeArray` instance method:

    >>> x = ivy.native_array([[1,1], [0,0], [1,1]])
    >>> y = x.nonzero()
    >>> print(y)
    tensor([[0,0],[0,1],[2,0],[2,1]])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1,1,1]), b=ivy.native_array([0]))
    >>> y = x.nonzero()
    >>> print(y)
    {
        a: (list[1], <class ivy.array.array.Array> shape=[3]),
        b: (list[1], <class ivy.array.array.Array> shape=[0])
    }

    >>> print(y.a)
    (ivy.array([0, 1, 2]),)

    >>> print(y.b)
    (ivy.array([]),)
    """
    return current_backend(x).nonzero(x)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def where(
    condition: Union[ivy.Array, ivy.NativeArray],
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with elements from x1 where condition is True, and elements from x2
        elsewhere.

    Functional Examples
    -------------------

    With `ivy.Array` input:

    >>> condition = ivy.array([[True, False], [True, True]])
    >>> x1 = ivy.array([[1, 2], [3, 4]])
    >>> x2 = ivy.array([[5, 6], [7, 8]])
    >>> res = ivy.where(condition, x1, x2)
    >>> print(res)
    ivy.array([[1,6],[3,4]])

    With `ivy.NativeArray` input:

    >>> condition = ivy.array([[True, False], [False, True]])
    >>> x1 = ivy.native_array([[1, 2], [3, 4]])
    >>> x2 = ivy.native_array([[5, 6], [7, 8]])
    >>> res = ivy.where(condition, x1, x2)
    >>> print(res)
    array([[1, 6], [7, 4]])

    With a mix of `ivy.Array` and `ivy.NativeArray` inputs:

    >>> x1 = ivy.array([[6, 13, 22, 7, 12], [7, 11, 16, 32, 9]])
    >>> x2 = ivy.native_array([[44, 20, 8, 35, 9], [98, 23, 43, 6, 13]])
    >>> res = ivy.where(((x1 % 2 == 0) & (x2 % 2 == 1)), x1, x2)
    >>> print(res)
    ivy.array([[ 44, 20, 8, 35, 12], [98, 23, 16, 6, 13]])

    With `ivy.Container` input:

    >>> x1 = ivy.Container(a=ivy.array([3, 1, 5]), b=ivy.array([2, 4, 6]))
    >>> x2 = ivy.Container(a=ivy.array([0, 7, 2]), b=ivy.array([3, 8, 5]))
    >>> res = ivy.where((x1.a > x2.a), x1, x2)
    >>> print(res)
    {
        a: ivy.array([3, 7, 5]),
        b: ivy.array([3, 8, 6])
    }

    With a mix of `ivy.Array` and `ivy.Container` inputs:

    >>> x1 = ivy.array([[1.1, 2, -3.6], [5, 4, 3.1]])
    >>> x2 = ivy.Container(a=ivy.array([0, 7, 2]),b=ivy.array([3, 8, 5]))
    >>> res = ivy.where((x1.b < x2.b), x1, x2)
    >>> print(res)
    {
        a: ivy.array([0, 2, -3.6]),
        b: ivy.array([3, 4, 3.1])
    }

    Instance Method Examples
    -------------------

    With `ivy.Array` input:

    >>> condition = ivy.array([[True, False], [True, True]])
    >>> x1 = ivy.array([[1, 2], [3, 4]])
    >>> x2 = ivy.array([[5, 6], [7, 8]])
    >>> res = x1.where(condition,x2)
    >>> print(res)
    ivy.array([[1, 6], [3, 4]])

    With `ivy.Container` input:

    >>> x1 = ivy.Container(a=ivy.array([3, 1, 5]), b=ivy.array([2, 4, 6]))
    >>> x2 = ivy.Container(a=ivy.array([0, 7, 2]), b=ivy.array([3, 8, 5]))
    >>> res = x1.where((x1.a > x2.a), x2)
    >>> print(res)
    {
        a: ivy.array([3, 7, 5]),
        b: ivy.array([2, 8, 6])
    }

    """
    return current_backend(x1).where(condition, x1, x2, out=out)


# Extra #
# ------#
