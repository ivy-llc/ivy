# global
from typing import Union, Optional, Tuple, List, Iterable, Sequence
from numbers import Number
from numpy.core.numeric import normalize_axis_tuple

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)


# Helpers #
# --------#


def _calculate_out_shape(axis, array_shape):
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_dims = len(axis) + len(array_shape)
    norm_axis = normalize_axis_tuple(axis, out_dims)
    shape_iter = iter(array_shape)
    out_shape = [
        1 if current_ax in norm_axis else next(shape_iter)
        for current_ax in range(out_dims)
    ]
    return out_shape


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def concat(
    xs: Union[
        Tuple[Union[ivy.Array, ivy.NativeArray]],
        List[Union[ivy.Array, ivy.NativeArray]],
    ],
    axis: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Casts an array to a specified type.

    Parameters
    ----------
    xs
        The input arrays must have the same shape, except in the dimension corresponding
        to axis (the first, by default).
    axis
        The axis along which the arrays will be joined. Default is -1.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The concatenated array.

    Examples
    --------
    >>> x = ivy.array([[1, 2], [3, 4]])
    >>> y = ivy.array([[5, 6]])
    >>> ivy.concat((x, y))
    ivy.array([[1, 2],
               [3, 4],
               [5, 6]])
    """
    return current_backend(xs[0]).concat(xs, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def expand_dims(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Union[int, Tuple[int], List[int]] = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Expands the shape of an array by inserting a new axis (dimension) of size one
    at the position specified by ``axis``

    Parameters
    ----------
    x
        input array.
    axis
        position in the expanded array where a new axis (dimension) of size one will be
        added. If array ``x`` has the rank of ``N``, the ``axis`` needs to be between
        ``[-N-1, N]``. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array with its dimension added by one in a given ``axis``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the `docstring # noqa
    <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 2]) #x.shape->(3,)
    >>> y = ivy.expand_dims(x) #y.shape->(1, 3)
    >>> print(y)
    ivy.array([[0, 1, 2]])

    >>> x = ivy.array([[0.5, -0.7, 2.4], \
                       [  1,    2,   3]]) #x.shape->(2, 3)
    >>> y = ivy.zeros((2, 1, 3))
    >>> ivy.expand_dims(x, axis=1, out=y) #y.shape->(2, 1, 3)
    >>> print(y)
    ivy.array([[[0.5, -0.7, 2.4]],
               [[ 1.,   2.,  3.]]])

    >>> x = ivy.array([[-1, -2], \
                       [ 3,  4]]) #x.shape->(2, 2)
    >>> ivy.expand_dims(x, axis=0, out=x) #x.shape->(1, 2, 2)
    >>> print(x)
    ivy.array([[[-1, -2],
                [3,  4]]])

    >>> x = ivy.array([[-1.1, -2.2, 3.3], \
                       [ 4.4,  5.5, 6.6]]) #x.shape->(2, 3)
    >>> y = ivy.expand_dims(x, axis=(0, -1)) #y.shape->(1, 2, 3, 1)
    >>> print(y)
    ivy.array([[[[-1.1],
                 [-2.2],
                 [ 3.3]],
                [[ 4.4],
                 [ 5.5],
                 [ 6.6]]]])

    >>> x = ivy.array([[-1.7, -3.2, 2.3], \
                       [ 6.3,  1.4, 5.7]]) #x.shape->(2, 3)
    >>> y = ivy.expand_dims(x, axis=[0, 1, -1]) ##y.shape->(1, 1, 2, 3, 1)
    >>> print(y)
    ivy.array([[[[[-1.7],
                  [-3.2],
                  [ 2.3]],
                 [[ 6.3],
                  [ 1.4],
                  [ 5.7]]]]])

    With one :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.expand_dims(x, axis=-1)
    >>> print(y)
    {
        a: ivy.array([[0.],
                      [1.],
                      [2.]]),
        b: ivy.array([[3.],
                      [4.],
                      [5.]])
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> container_axis = ivy.Container(a=0, b=1)
    >>> y = ivy.expand_dims(x, axis=container_axis)
    >>> print(y)
    {
        a: ivy.array([[0., 1., 2.]]),
        b: ivy.array([[3.],
                      [4.],
                      [5.]])
    }
    """
    return current_backend(x).expand_dims(x, axis)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def flip(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Reverses the order of elements in an array along the given axis. The shape of the
    array must be preserved.

    Parameters
    ----------
    x
        input array.
    axis
        axis (or axes) along which to flip. If ``axis`` is ``None``, the function must
        flip all input array axes. If ``axis`` is negative, the function must count from
        the last dimension. If provided more than one axis, the function must flip only
        the specified axes. Default  ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an output array having the same data type and shape as ``x`` and whose elements,
        relative to ``x``, are reordered.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.flip.html>`_ # noqa
    in the standard. The descriptions above assume an array input for simplicity, but
    the method also accepts :code:`ivy.Container` instances in place of
    :code:`ivy.Array` or :code:`ivy.NativeArray` instances, as shown in the type hints
    and also the examples below.

    Functional Examples
    -------------------
    With :code:`ivy.Array` input:
    >>> x = ivy.array([3, 4, 5])
    >>> y = ivy.flip(x)
    >>> print(y)
    ivy.array([5, 4, 3])

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> y = ivy.zeros((3, 3))
    >>> ivy.flip(x, out=y)
    >>> print(y)
    ivy.array([[6, 5, 4],
               [3, 2, 1]])

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> y = ivy.zeros((3, 3))
    >>> ivy.flip(x, axis=0, out=y)
    >>> print(y)
    ivy.array([[4, 5, 6],
               [1, 2, 3]])

    >>> x = ivy.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
    >>> ivy.flip(x, axis=[0, 1], out=x)
    >>> print(x)
    ivy.array([[[10,11,12],[7,8,9]],[[4,5,6],[1,2,3]]])

    >>> x = ivy.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
    >>> ivy.flip(x, axis=(2, 1), out=x)
    >>> print(x)
    ivy.array([[[ 6,  5,  4],
                [ 3,  2,  1]],
               [[12, 11, 10],
                [ 9,  8,  7]]])

    With :code:`ivy.NativeArray` input:
    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.flip(x)
    >>> print(y)
    ivy.array([2., 1., 0.])

    With :code:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                      b=ivy.array([3., 4., 5.]))
    >>> y = ivy.flip(x)
    >>> print(y)
    {
        a: ivy.array([2., 1., 0.]),
        b: ivy.array([5., 4., 3.])
    }

    Instance Method Examples
    ------------------------
    Using :code:`ivy.Array` instance method:
    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.flip()
    >>> print(y)
    ivy.array([2., 1., 0.])

    Using :code:`ivy.Container` instance method:
    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.flip()
    >>> print(y)
    {
        a: ivy.array([2., 1., 0.]),
        b: ivy.array([5., 4., 3.])
    }

    """
    return current_backend(x).flip(x, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def permute_dims(
    x: Union[ivy.Array, ivy.NativeArray],
    axes: Tuple[int, ...],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Permutes the axes (dimensions) of an array x.

    Parameters
    ----------
    x
        input array.
    axes
        tuple containing a permutation of (0, 1, ..., N-1) where N is the number of axes
        (dimensions) of x.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the axes permutation. The returned array must have the same
        data type as x.

    """
    return current_backend(x).permute_dims(x, axes, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def reshape(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    *,
    copy: Optional[bool] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    x
        Input array to be reshaped.
    shape
        The new shape should be compatible with the original shape. One shape dimension
        can be -1. In this case, the value is inferred from the length of the array and
        remaining dimensions.
    copy
        boolean indicating whether or not to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Reshaped array.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([[0., 1., 2.], \
                       [3., 4., 5.]])
    >>> y = ivy.reshape(x,(3,2))
    >>> print(y)
    ivy.array([[0., 1.],
               [2., 3.],
               [4., 5.]])


    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[0., 1., 2.],[3., 4., 5.]])
    >>> y = ivy.reshape(x,(2,3))
    >>> print(y)
    ivy.array([[0., 1., 2.],
               [3., 4., 5.]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]), \
                          b=ivy.array([0, 1, 2, 3, 4, 5]))
    >>> y = ivy.reshape(x,(2,3))
    >>> print(y)
    {
        a: ivy.array([[0, 1, 2],
                      [3, 4, 5]]),
        b: ivy.array([[0, 1, 2],
                      [3, 4, 5]])
    }

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[0, 1, 2, 3]])
    >>> y = ivy.reshape(x, (2, 2))
    >>> print(y)
    ivy.array([[0, 1],
               [2, 3]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), b=ivy.array([[3., 4., 5.]]))
    >>> y = ivy.reshape(x, (-1, 1))
    >>> print(y)
    {
        a: ivy.array([[0.],
                      [1.],
                      [2.]]),
        b: ivy.array([[3.],
                      [4.],
                      [5.]])
    }

    """
    return current_backend(x).reshape(x, shape, copy, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def roll(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    shift: Union[int, Sequence[int]],
    axis: Optional[Union[int, Sequence[int]]] = None,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.Container]:
    """Rolls array elements along a specified axis. Array elements that roll beyond the
    last position are re-introduced at the first position. Array elements that roll
    beyond the first position are re-introduced at the last position.

    Parameters
    ----------
    x
        input array.
    shift
        number of places by which the elements are shifted. If ``shift`` is a tuple,
        then ``axis`` must be a tuple of the same size, and each of the given axes must
        be shifted by the corresponding element in ``shift``. If ``shift`` is an ``int``
        and ``axis`` a tuple, then the same ``shift`` must be used for all specified
        axes. If a shift is positive, then array elements must be shifted positively
        (toward larger indices) along the dimension of ``axis``. If a shift is negative,
        then array elements must be shifted negatively (toward smaller indices) along
        the dimension of ``axis``.
    axis
        axis (or axes) along which elements to shift. If ``axis`` is ``None``, the array
        must be flattened, shifted, and then restored to its original shape.
        Default ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an output array having the same data type as ``x`` and whose elements, relative
        to ``x``, are shifted.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.roll.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    >>> x = ivy.array([[0., 1., 2.], \
                       [3., 4., 5.]])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.roll(x, 2, -1, out=y)
    >>> print(y)
    ivy.array([[1., 2., 0.],
                [4., 5., 3.]])

    >>> x = ivy.array([[[0., 0.], [1., 3.], [2., 6.]], \
                       [[3., 9.], [4., 12.], [5., 15.]]])
    >>> ivy.roll(x, (1, -1), (0, 2), out=x)
    >>> print(x)
    ivy.array([[[ 9., 3.],
                [12., 4.],
                [15., 5.]],
               [[ 0., 0.],
                [ 3., 1.],
                [ 6., 2.]]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2.])
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    ivy.array([2., 0., 1.])


    With one :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.roll(x, 1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> shift = ivy.Container(a=1, b=-1)
    >>> y = ivy.roll(x, shift)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([4., 5., 3.])
    }

    Instance Method Examples
    ------------------------
    >>> x = ivy.array([[0., 1., 2.], \
                       [3., 4., 5.]])
    >>> y = x.roll(2, -1)
    >>> print(y)
    ivy.array([[1., 2., 0.],
                [4., 5., 3.]])

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = x.roll(1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }
    """
    return current_backend(x).roll(x, shift, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def squeeze(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Removes singleton dimensions (axes) from ``x``.

    Parameters
    ----------
    x
        input array.
    axis
        axis (or axes) to squeeze. If a specified axis has a size greater than one, a
        ``ValueError`` must be raised.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an output array having the same data type and elements as ``x``.


    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([[[0, 1], [2, 3]]])
    >>> y = ivy.squeeze(x)
    >>> print(y)
    ivy.array([[0, 1], [2, 3]])

    >>> x = ivy.array([[[[1, 2, 3]], [[4, 5, 6]]]])
    >>> ivy.squeeze(x, axis=2)
    >>> print(y)
    ivy.array([[0,1],[2,3]])

    >>> x = ivy.array([[[0], [1], [2]]])
    >>> y = ivy.squeeze(x)
    >>> print(y)
    ivy.array([0, 1, 2])

    >>> y = ivy.squeeze(x, axis=0)
    >>> print(y)
    ivy.array([[0],
           [1],
           [2]])

    >>> y = ivy.squeeze(x, axis=2)
    >>> print(y)
    ivy.array([[0, 1, 2]])

    >>> y = ivy.squeeze(x, axis=(0, 2))
    >>> print(y)
    ivy.array([0, 1, 2])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 2])
    >>> y = ivy.squeeze(x)
    >>> print(y)
    ivy.array([0, 1, 2])

    >>> x = ivy.native_array([[[3]]])
    >>> y = ivy.squeeze(x, 2)
    >>> print(y)
    ivy.array([[3]])

    >>> x = ivy.native_array([0])
    >>> print(ivy.squeeze(x, 0))
    ivy.array(0)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.squeeze(x)
    >>> print(y)
    {
        a: ivy.array([0., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }
    """
    return current_backend(x).squeeze(x, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def stack(
    arrays: Union[
        Tuple[ivy.Array], List[ivy.Array], Tuple[ivy.NativeArray], List[ivy.NativeArray]
    ],
    axis: int = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays
        input arrays to join. Each array must have the same shape.
    axis
        axis along which the arrays will be joined. Providing an ``axis`` specifies the
        index of the new axis in the dimensions of the result. For example, if ``axis``
        is ``0``, the new axis will be the first dimension and the output array will
        have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the
        second dimension and the output array will have shape ``(A, N, B, C)``; and, if
        ``axis`` is ``-1``, the new axis will be the last dimension and the output array
        will have shape ``(A, B, C, N)``. A valid ``axis`` must be on the interval
        ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If
        provided an ``axis`` outside of the required interval, the function must raise
        an exception. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an output array having rank ``N+1``, where ``N`` is the rank (number of
        dimensions) of ``x``. If the input arrays have different data types, normal
        ref:`type-promotion` must apply. If the input arrays have the same data type,
        the output array must have the same data type as the input arrays.
        .. note::
           This specification leaves type promotion between data type families (i.e.,
           ``intxx`` and ``floatxx``) unspecified.

    """
    return current_backend(arrays).stack(arrays, axis, out=out)


# Extra #
# ------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def clip(
    x: Union[ivy.Array, ivy.NativeArray],
    x_min: Union[Number, ivy.Array, ivy.NativeArray],
    x_max: Union[Number, ivy.Array, ivy.NativeArray],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges
    (element-wise). For example, if an interval of [0, 1] is specified, values smaller
    than 0 become 0, and values larger than 1 become 1. Minimum value needs to smaller
    or equal to maximum value to return correct results.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    x_min
        Minimum value.
    x_max
        Maximum value.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the elements of x, but where values < x_min are replaced with
        x_min, and those > x_max with x_max.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> y = ivy.clip(x, 1., 5.)
    >>> print(y)
    ivy.array([1., 1., 2., 3., 4., 5., 5., 5., 5., 5.])

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> y = ivy.zeros_like(x)
    >>> ivy.clip(x, 2., 7., out=y)
    >>> print(y)
    ivy.array([2., 2., 2., 3., 4., 5., 6., 7., 7., 7.])

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> x_min = ivy.array([3., 3., 1., 0., 2., 3., 4., 0., 4., 4.])
    >>> x_max = ivy.array([5., 4., 3., 3., 5., 7., 8., 3., 8., 8.])
    >>> y = ivy.clip(x, x_min, x_max)
    >>> print(y)
    ivy.array([3., 3., 2., 3., 4., 5., 6., 3., 8., 8.])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> x_min = ivy.native_array([3., 3., 1., 0., 2., 3., 4., 2., 4., 4.])
    >>> x_max = ivy.native_array([5., 4., 3., 3., 5., 7., 8., 3., 8., 8.])
    >>> y = ivy.clip(x, x_min, x_max)
    >>> print(y)
    ivy.array([3., 3., 2., 3., 4., 5., 6., 3., 8., 8.])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> x_min = ivy.native_array([3., 3., 1., 0., 2., 3., 4., 2., 4., 4.])
    >>> x_max = ivy.native_array([5., 4., 3., 3., 5., 7., 8., 3., 8., 8.])
    >>> y = ivy.clip(x, x_min, x_max)
    >>> print(y)
    ivy.array([3., 3., 2., 3., 4., 5., 6., 3., 8., 8.])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> y = ivy.clip(x, 1., 5.)
    >>> print(y)
    {
        a: ivy.array([1., 1., 2.]),
        b: ivy.array([3., 4., 5.])
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> x_min = ivy.Container(a=0, b=-3)
    >>> x_max = ivy.Container(a=1, b=-1)
    >>> y = ivy.clip(x, x_min,x_max)
    >>> print(y)
    {
        a: ivy.array([0., 1., 1.]),
        b: ivy.array([-1., -1., -1.])
    }

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> x_min = ivy.array([3., 0., 1])
    >>> x_max = ivy.array([5., 4., 3.])
    >>> y = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([3., 4., 5.]))
    >>> z = ivy.clip(y, x_min, x_max)
    >>> print(z)
    {
        a: ivy.array([3., 1., 2.]),
        b: ivy.array([3., 4., 3.])
    }

    """
    assert ivy.all(ivy.less(x_min, x_max))
    res = current_backend(x).clip(x, x_min, x_max)
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def constant_pad(
    x: Union[ivy.Array, ivy.NativeArray],
    pad_width: Iterable[Tuple[int]],
    value: Number = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Pads an array with a constant value.

    Parameters
    ----------
    x
        Input array to pad.
    pad_width
        Number of values padded to the edges of each axis.
        Specified as ((before_1, after_1), … (before_N, after_N)), where N is number of
        axes of x.
    value
        The constant value to pad the array with.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Padded array of rank equal to x with shape increased according to pad_width.

    """
    return current_backend(x).constant_pad(x, pad_width, value, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def repeat(
    x: Union[ivy.Array, ivy.NativeArray],
    repeats: Union[int, Iterable[int]],
    axis: int = None,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Repeat values along a given dimension.

    Parameters
    ----------
    x
        Input array.
    repeats
        The number of repetitions for each element. repeats is broadcast to fit the
        shape of the given axis.
    axis
        The axis along which to repeat values. By default, use the flattened input
        array, and return a flat output array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The repeated output array.


    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([3, 4, 5])
    >>> y= ivy.repeat(x, 2)
    >>> print(y)
    ivy.array([3, 3, 4, 4, 5, 5])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[1, 2, 3], [4, 5, 6]])
    >>> y = ivy.repeat(x, [1, 2], axis=0)
    >>> print(y)
    ivy.array([[1, 2, 3],
               [4, 5, 6],
               [4, 5, 6]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                          b=ivy.array([0., 1., 2.]))
    >>> y = ivy.repeat(x, 2, axis=0)
    >>> print(y)
    {
        a: ivy.array([0., 0., 1., 1., 2., 2.]),
        b: ivy.array([0., 0., 1., 1., 2., 2.])
    }
    """
    return current_backend(x).repeat(x, repeats, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def split(
    x: Union[ivy.Array, ivy.NativeArray],
    num_or_size_splits: Optional[Union[int, Iterable[int]]] = None,
    axis: Optional[int] = 0,
    with_remainder: Optional[bool] = False,
) -> ivy.Array:
    """Splits an array into multiple sub-arrays.

    Parameters
    ----------
    x
        array to be divided into sub-arrays.
    num_or_size_splits
        Number of equal arrays to divide the array into along the given axis if an
        integer. The size of each split element if a sequence of integers. Default is to
        divide into as many 1-dimensional arrays as the axis dimension.
    axis
        The axis along which to split, default is 0.
    with_remainder
        If the tensor does not split evenly, then store the last remainder entry.
        Default is False.

    Returns
    -------
    ret
        A list of sub-arrays.

    Functional Examples
    -------------------

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.split(x)
    >>> print(y)
    [ivy.array([1]),ivy.array([2]),ivy.array([3])]

    >>> x = ivy.array([[3, 2, 1], [4, 5, 6]])
    >>> y = ivy.split(x, 2, 1, False)
    >>> print(y)
    [ivy.array([[3,2],[4,5]]),ivy.array([[1],[6]])]

    >>> x = ivy.array([4, 6, 5, 3])
    >>> y = ivy.split(x, [1, 2], 0, True)
    >>> print(y)
    ivy.array([[4], [6, 5, 3]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([7, 8, 9])
    >>> y = ivy.split(x)
    >>> print(y)
    [ivy.array([7]),ivy.array([8]),ivy.array([9])]

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([10, 45, 2]))
    >>> y = ivy.split(x)
    >>> print(y)
    {a:(list[3],<classivy.array.Array>shape=[1])}

    Instance Method Examples
    ------------------------
    >>> x = ivy.array([4, 6, 5, 3])
    >>> y = x.split()
    >>> print(y)
    [ivy.array([4]),ivy.array([6]),ivy.array([5]),ivy.array([3])]

    >>> x = ivy.Container(a=ivy.array([2, 5, 9]))
    >>> y = x.split()
    >>> print(y)
    {
        a: ivy.array([[2], [5], [9]])
    }
    """
    return current_backend(x).split(x, num_or_size_splits, axis, with_remainder)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def swapaxes(
    x: Union[ivy.Array, ivy.NativeArray],
    axis0: int,
    axis1: int,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Interchange two axes of an array.

    Parameters
    ----------
    x
        Input array.
    axis0
        First axis to be swapped.
    axis1
        Second axis to be swapped.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        x with its axes permuted.

    Functional Examples
    -------------------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([[0, 1, 2]])
    >>> y = ivy.swapaxes(x, 0, 1)
    >>> print(y)
    ivy.array([[0],
               [1],
               [2]])

    >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> y = ivy.swapaxes(x, 0, 1)
    >>> print(y)
    ivy.array([[[0, 1],
                [4, 5]],
               [[2, 3],
                [6, 7]]])

    >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> y = ivy.swapaxes(x, 0, 2)
    >>> print(y)
    ivy.array([[[0, 4],
                [2, 6]],
               [[1, 5],
                [3, 7]]])

    >>> x = ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> y = ivy.swapaxes(x, 1, 2)
    >>> print(y)
    ivy.array([[[0, 2],
                [1, 3]],
               [[4, 6],
                [5, 7]]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[0, 1, 2]])
    >>> y = ivy.swapaxes(x, 0, 1)
    >>> print(y)
    ivy.array([[0],
               [1],
               [2]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), b=ivy.array([[3., 4., 5.]]))
    >>> y = ivy.swapaxes(x, 0, 1)
    >>> print(y)
    {
        a: ivy.array([[0.],
                      [1.],
                      [2.]]),
        b: ivy.array([[3.],
                      [4.],
                      [5.]])
    }

    Instance Method Examples
    ------------------------
    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([[0., 1., 2.]])
    >>> y = x.swapaxes(0, 1)
    >>> print(y)
    ivy.array([[0.],
               [1.],
               [2.]])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), b=ivy.array([[3., 4., 5.]]))
    >>> y = x.swapaxes(0, 1)
    >>> print(y)
    {
        a: ivy.array([[0.],
                      [1.],
                      [2.]]),
        b: ivy.array([[3.],
                      [4.],
                      [5.]])
    }

    """
    return current_backend(x).swapaxes(x, axis0, axis1, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def tile(
    x: Union[ivy.Array, ivy.NativeArray],
    reps: Iterable[int],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Constructs an array by repeating x the number of times given by reps.

    Parameters
    ----------
    x
        Input array.
    reps
        The number of repetitions of x along each axis.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The tiled output array.

    """
    return current_backend(x).tile(x, reps, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def zero_pad(
    x: Union[ivy.Array, ivy.NativeArray],
    pad_width: Iterable[Tuple[int]],
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Pads an array with zeros.

    Parameters
    ----------
    x
        Input array to pad.
    pad_width
        Number of values padded to the edges of each axis. Specified as
        ((before_1, after_1), … (before_N, after_N)), where N is number of axes of x.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Padded array of rank equal to x with shape increased according to pad_width.

    """
    return current_backend(x).zero_pad(x, pad_width, out=out)
