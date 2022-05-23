# global
from typing import Union, Optional, Tuple, List, Iterable
from numbers import Number

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework


# Array API Standard #
# -------------------#


def roll(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    out: Optional[Union[ivy.Array, ivy.Container]] = None,
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


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.roll.html>`_ # noqa
    in the standard. The descriptions above assume an array input for simplicity, but
    the method also accepts :code:`ivy.Container` instances in place of
    :code:`ivy.Array` or :code:`ivy.NativeArray` instances, as shown in the type hints
    and also the examples below.

    Functional Examples
    -------------------

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

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                      b=ivy.array([3., 4., 5.]))

    >>> y = ivy.roll(x, 1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0., 1., 2.])
    >>> y = x.roll(1)
    >>> print(y)
    ivy.array([2., 0., 1.])

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> y = x.roll(1)
    >>> print(y)
    {
        a: ivy.array([2., 0., 1.]),
        b: ivy.array([5., 3., 4.])
    }

    """
    return _cur_framework(x).roll(x, shift, axis, out)


def squeeze(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Union[int, Tuple[int, ...]],
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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


    Examples
    --------
    >>> x = ivy.array([[[0, 1], [2, 3]]])
    >>> print(x.shape)
    (1, 2, 2)

    >>> print(ivy.squeeze(x, axis=0).shape)
    (2, 2)

    """
    return _cur_framework(x).squeeze(x, axis, out)


def flip(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[Union[int, Tuple[int], List[int]]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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

    """
    return _cur_framework(x).flip(x, axis, out)


def expand_dims(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Expands the shape of an array by inserting a new axis with the size of one. This
    new axis will appear at the ``axis`` position in the expanded array shape.

    Parameters
    ----------
    x
        input array.
    axis
        position in the expanded array where a new axis (dimension) of size one will be
        added. If array ``x`` has the rank of ``N``, the ``axis`` need to be between
        ``[-N-1, N]``. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array with its dimension added by one in a given ``axis``.

    Examples
    --------
    >>> x = ivy.array([[0, 1], [1, 0]])
    >>> y = ivy.expand_dims(x)
    >>> print(y)
    ivy.array([[[0, 1],
                [1, 0]]])
                
    >>> print(x.shape, y.shape)
    (2, 2) (1, 2, 2)

    """
    return _cur_framework(x).expand_dims(x, axis, out)


def permute_dims(
    x: Union[ivy.Array, ivy.NativeArray],
    axes: Tuple[int, ...],
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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
    return _cur_framework(x).permute_dims(x, axes, out)


def stack(
    arrays: Union[
        Tuple[ivy.Array], List[ivy.Array], Tuple[ivy.NativeArray], List[ivy.NativeArray]
    ],
    axis: int = 0,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
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

    Examples
    -------
    The function joins arrays along first dimension by default:

    >>> x_1 = ivy.array([1, 2, 3])
    >>> x_2 = ivy.array([4, 5, 6])
    >>> y = ivy.stack([x_1, x_2])

    >>> y
    ivy.array([[1, 2, 3],
               [4, 5, 6]])
    >>> y.shape
    (2, 3)

    if ``axis=1`` it will stack arrays along the second dimension:

    >>> x_1 = ivy.array([1, 2, 3])
    >>> x_2 = ivy.array([4, 5, 6])
    >>> y = ivy.zero((3,2))

    >>> ivy.stack((x_1, x_2), axis=1, out=y)
    ivy.array([[1., 4.],
               [2., 5.],
               [3., 6.]], dtype=float32)
    >>> y.shape
    (3, 2)

    """
    return _cur_framework(arrays).stack(arrays, axis, out)


def reshape(
    x: Union[ivy.Array, ivy.NativeArray],
    shape: Tuple[int, ...],
    copy: Optional[bool] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Gives a new shape to an array without changing its data.

    Parameters
    ----------
    x
        Tensor to be reshaped.
    newshape
        The new shape should be compatible with the original shape. One shape dimension
        can be -1. In this case, the value is inferred from the length of the array and
        remaining dimensions.

    Returns
    -------
    ret
        Reshaped array.

    Examples
    --------
    >>> x = ivy.array([[1,2,3], [4,5,6]])
    >>> y = ivy.reshape(x, (3,2))
    >>> print(y)
    ivy.array([[1, 2],
               [3, 4],
               [5, 6]])

    """
    return _cur_framework(x).reshape(x, shape, copy, out)


def concat(
    xs: Union[
        Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
    ],
    axis: Optional[int] = 0,
    out: Optional[Union[ivy.Array, ivy.NativeArray, ivy.Container]] = None,
) -> Union[ivy.Array, ivy.Container]:
    """Casts an array to a specified type.

    Parameters
    ----------
    xs
        The input arrays must have the same shape, except in the dimension corresponding
        to axis (the first, by default).
    axis
        The axis along which the arrays will be joined. Default is -1.

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
    return _cur_framework(xs[0]).concat(xs, axis, out)


# Extra #
# ------#


def split(
    x: Union[ivy.Array, ivy.NativeArray],
    num_or_size_splits: Union[int, Iterable[int]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Splits an array into multiple sub-arrays.

    Parameters
    ----------
    x
        Tensor to be divided into sub-arrays.
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

    """
    return _cur_framework(x).split(x, num_or_size_splits, axis, with_remainder)


def repeat(
    x: Union[ivy.Array, ivy.NativeArray],
    repeats: Union[int, Iterable[int]],
    axis: int = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
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

    Returns
    -------
    ret
        The repeated output array.

    """
    return _cur_framework(x).repeat(x, repeats, axis, out)


def tile(
    x: Union[ivy.Array, ivy.NativeArray],
    reps: Iterable[int],
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Constructs an array by repeating x the number of times given by reps.

    Parameters
    ----------
    x
        Input array.
    reps
        The number of repetitions of x along each axis.

    Returns
    -------
    ret
        The tiled output array.

    """
    return _cur_framework(x).tile(x, reps, out)


def constant_pad(
    x: Union[ivy.Array, ivy.NativeArray],
    pad_width: Iterable[Tuple[int]],
    value: Number = 0,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
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

    Returns
    -------
    ret
        Padded array of rank equal to x with shape increased according to pad_width.

    """
    return _cur_framework(x).constant_pad(x, pad_width, value, out)


def zero_pad(
    x: Union[ivy.Array, ivy.NativeArray],
    pad_width: Iterable[Tuple[int]],
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

    Returns
    -------
    ret
        Padded array of rank equal to x with shape increased according to pad_width.

    """
    return _cur_framework(x).zero_pad(x, pad_width, out)


def swapaxes(
    x: Union[ivy.Array, ivy.NativeArray],
    axis0: int,
    axis1: int,
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

    Returns
    -------
    ret
        x with its axes permuted.

    """
    return _cur_framework(x).swapaxes(x, axis0, axis1, out)


def clip(
    x: Union[ivy.Array, ivy.NativeArray],
    x_min: Union[Number, Union[ivy.Array, ivy.NativeArray]],
    x_max: Union[Number, Union[ivy.Array, ivy.NativeArray]],
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges
    (element-wise). For example, if an interval of [0, 1] is specified, values smaller
    than 0 become 0, and values larger than 1 become 1.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    x_min
        Minimum value.
    x_max
        Maximum value.

    Returns
    -------
    ret
        An array with the elements of x, but where values < x_min are replaced with
        x_min, and those > x_max with x_max.

    """
    return _cur_framework(x).clip(x, x_min, x_max, out)
