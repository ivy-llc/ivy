# global
from typing import (
    Optional,
    Union,
    Tuple,
    Iterable,
    Sequence,
    Callable,
    Any,
    Literal,
    List,
)
from numbers import Number
from functools import partial
import math

# local
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    handle_partial_mixed_function,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_view,
    inputs_to_ivy_arrays,
    handle_array_function,
    handle_device_shifting,
)
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@handle_array_like_without_promotion
@handle_view
@inputs_to_ivy_arrays
@handle_array_function
def flatten(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Flattens input by reshaping it into a one-dimensional tensor. If start_dim or
    end_dim are passed, only dimensions starting with start_dim and ending with end_dim
    are flattened. The order of elements in input is unchanged.

    Parameters
    ----------
    x
        input array to flatten.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    start_dim
        first dim to flatten. If not set, defaults to 0.
    end_dim
        last dim to flatten. If not set, defaults to -1.
    order
        Read the elements of the input container using this index order,
        and place the elements into the reshaped array using this index order.
        ‘C’ means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first axis index
        changing slowest.
        ‘F’ means to read / write the elements using Fortran-like index order, with
        the first index changing fastest, and the last index changing slowest.
        Note that the ‘C’ and ‘F’ options take no account of the memory layout
        of the underlying array, and only refer to the order of indexing.
        Default order is 'C'
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        the flattened array over the specified dimensions.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[1,2], [3,4]])
    >>> ivy.flatten(x)
    ivy.array([1, 2, 3, 4])

    >>> x = ivy.array([[1,2], [3,4]])
    >>> ivy.flatten(x, order='F')
    ivy.array([1, 3, 2, 4])

    >>> x = ivy.array(
        [[[[ 5,  5,  0,  6],
         [17, 15, 11, 16],
         [ 6,  3, 13, 12]],

        [[ 6, 18, 10,  4],
         [ 5,  1, 17,  3],
         [14, 14, 18,  6]]],


       [[[12,  0,  1, 13],
         [ 8,  7,  0,  3],
         [19, 12,  6, 17]],

        [[ 4, 15,  6, 15],
         [ 0,  5, 17,  9],
         [ 9,  3,  6, 19]]],


       [[[17, 13, 11, 16],
         [ 4, 18, 17,  4],
         [10, 10,  9,  1]],

        [[19, 17, 13, 10],
         [ 4, 19, 16, 17],
         [ 2, 12,  8, 14]]]]
         )
    >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
    ivy.array(
        [[[ 5,  5,  0,  6],
          [17, 15, 11, 16],
          [ 6,  3, 13, 12],
          [ 6, 18, 10,  4],
          [ 5,  1, 17,  3],
          [14, 14, 18,  6]],

         [[12,  0,  1, 13],
          [ 8,  7,  0,  3],
          [19, 12,  6, 17],
          [ 4, 15,  6, 15],
          [ 0,  5, 17,  9],
          [ 9,  3,  6, 19]],

         [[17, 13, 11, 16],
          [ 4, 18, 17,  4],
          [10, 10,  9,  1],
          [19, 17, 13, 10],
          [ 4, 19, 16, 17],
          [ 2, 12,  8, 14]]]))
    """
    if copy:
        x = ivy.copy_array(x)
    if x.shape == ():
        x = ivy.reshape(x, (1, -1))[0, :]
    if start_dim == end_dim:
        return ivy.inplace_update(out, x) if ivy.exists(out) else x
    if start_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            "Dimension out of range (expected to be in range of"
            f" {[-len(x.shape), len(x.shape) - 1]}, but got {start_dim}"
        )
    if end_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            "Dimension out of range (expected to be in range of"
            f" {[-len(x.shape), len(x.shape) - 1]}, but got {end_dim}"
        )
    if start_dim < 0:
        start_dim = len(x.shape) + start_dim
    if end_dim < 0:
        end_dim = len(x.shape) + end_dim
    c = 1
    for i in range(start_dim, end_dim + 1):
        c *= x.shape[i]
    lst = [c]
    if start_dim != 0:
        for i in range(0, start_dim):
            lst.insert(i, x.shape[i])
    for i in range(end_dim + 1, len(x.shape)):
        lst.insert(i, x.shape[i])
    return ivy.reshape(x, tuple(lst), order=order, out=out)


flatten.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def moveaxis(
    a: Union[ivy.Array, ivy.NativeArray],
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Move axes of an array to new positions..

    Parameters
    ----------
    a
        The array whose axes should be reordered.
    source
        Original positions of the axes to move. These must be unique.
    destination
        Destination positions for each of the original axes.
        These must also be unique.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with moved axes. This array is a view of the input array.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.zeros((3, 4, 5))
    >>> ivy.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> ivy.moveaxis(x, -1, 0).shape
    (5, 3, 4)
    """
    return ivy.current_backend().moveaxis(a, source, destination, copy=copy, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def heaviside(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the Heaviside step function for each element in x1.

    Parameters
    ----------
    x1
        input array.
    x2
        values to use where x1 is zero.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        output array with element-wise Heaviside step function of x1.
        This is a scalar if both x1 and x2 are scalars.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([-1.5, 0, 2.0])
    >>> x2 = ivy.array([0.5])
    >>> ivy.heaviside(x1, x2)
    ivy.array([0.0000, 0.5000, 1.0000])

    >>> x1 = ivy.array([-1.5, 0, 2.0])
    >>> x2 = ivy.array([1.2, -2.0, 3.5])
    >>> ivy.heaviside(x1, x2)
    ivy.array([0., -2., 1.])
    """
    return ivy.current_backend().heaviside(x1, x2, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def flipud(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Flip array in the up/down direction. Flip the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m
        The array to be flipped.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array corresponding to input array with elements
        order reversed along axis 0.

    Examples
    --------
    >>> m = ivy.diag([1, 2, 3])
    >>> ivy.flipud(m)
    ivy.array([[ 0.,  0.,  3.],
        [ 0.,  2.,  0.],
        [ 1.,  0.,  0.]])
    """
    return ivy.current_backend().flipud(m, copy=copy, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def vstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Stack arrays in sequence vertically (row wise).

    Parameters
    ----------
    arrays
        Sequence of arrays to be stacked.

    Returns
    -------
    ret
        The array formed by stacking the given arrays.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([2, 3, 4])
    >>> ivy.vstack((x, y))
    ivy.array([[1, 2, 3],
           [2, 3, 4]])
    >>> ivy.vstack((x, y, x, y))
    ivy.array([[1, 2, 3],
               [2, 3, 4],
               [1, 2, 3],
               [2, 3, 4]])

    >>> y = [ivy.array([[5, 6]]), ivy.array([[7, 8]])]
    >>> print(ivy.vstack(y))
    ivy.array([[5, 6],
               [7, 8]])
    """
    return ivy.current_backend().vstack(arrays, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def hstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """
    Stack arrays in sequence horizotally (column wise).

    Parameters
    ----------
    arrays
        Sequence of arrays to be stacked.

    Returns
    -------
    ret
        The array formed by stacking the given arrays.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([2, 3, 4])
    >>> ivy.hstack((x, y))
    ivy.array([1, 2, 3, 2, 3, 4])
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([0, 0, 0])
    >>> ivy.hstack((x, y, x))
    ivy.array([1, 2, 3, 0, 0, 0, 1, 2, 3])
    >>> y = [ivy.array([[5, 6]]), ivy.array([[7, 8]])]
    >>> print(ivy.hstack(y))
    ivy.array([[5, 6, 7, 8]])
    """
    return ivy.current_backend().hstack(arrays, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_view
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def rot90(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Rotate an array by 90 degrees in the plane specified by axes. Rotation direction is
    from the first towards the second axis.

    Parameters
    ----------
    m
        Input array of two or more dimensions.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    k
        Number of times the array is rotated by 90 degrees.
    axes
        The array is rotated in the plane defined by the axes. Axes must be
        different.
    out
        optional output container, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        A rotated view of m.

    Examples
    --------
    With :code:`ivy.Array` input:
    >>> m = ivy.array([[1,2], [3,4]])
    >>> ivy.rot90(m)
    ivy.array([[2, 4],
           [1, 3]])
    >>> m = ivy.array([[1,2], [3,4]])
    >>> ivy.rot90(m, k=2)
    ivy.array([[4, 3],
           [2, 1]])
    >>> m = ivy.array([[[0, 1],\
                        [2, 3]],\
                       [[4, 5],\
                        [6, 7]]])
    >>> ivy.rot90(m, k=2, axes=(1,2))
    ivy.array([[[3, 2],
            [1, 0]],

           [[7, 6],
            [5, 4]]])
    With :code:`ivy.NativeArray` input:
    >>> m = ivy.native_array([[1,2], [3,4]])
    >>> ivy.rot90(m)
    ivy.array([[2, 4],
           [1, 3]])
    >>> m = ivy.native_array([[1,2], [3,4]])
    >>> ivy.rot90(m, k=2)
    ivy.array([[4, 3],
           [2, 1]])
    >>> m = ivy.native_array([[[0, 1],\
                               [2, 3]],\
                              [[4, 5],\
                               [6, 7]]])
    >>> ivy.rot90(m, k=2, axes=(1,2))
    ivy.array([[[3, 2],
            [1, 0]],

           [[7, 6],
            [5, 4]]])
    """
    return ivy.current_backend(m).rot90(m, copy=copy, k=k, axes=axes, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def top_k(
    x: Union[ivy.Array, ivy.NativeArray],
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[tuple] = None,
) -> Tuple[ivy.Array, ivy.NativeArray]:
    """
    Return the `k` largest elements of the given input array along a given axis.

    Parameters
    ----------
    x
        The array to compute top_k for.
    k
        Number of top elements to retun must not exceed the array size.
    axis
        The axis along which we must return the top elements default value is 1.
    largest
        If largest is set to False we return k smallest elements of the array.
    sorted
        If sorted is set to True we return the elements in sorted order.
    out:
        Optional output tuple, for writing the result to. Must have two arrays inside,
        with a shape that the returned tuple broadcast to.

    Returns
    -------
    ret
        A named tuple with values and indices of top k elements.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([2., 1., -3., 5., 9., 0., -4])
    >>> y = ivy.top_k(x, 2)
    >>> print(y)
    top_k(values=ivy.array([9., 5.]), indices=ivy.array([4, 3]))

    >>> x = ivy.array([[-2., 3., 4., 0.], [-8., 0., -1., 2.]])
    >>> y = ivy.top_k(x, 2, axis=1, largest=False)
    >>> print(y)
    top_k(values=ivy.array([[-2.,  0.],[-8., -1.]]),
    ...   indices=ivy.array([[0, 3],[0, 2]]))

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([2., 1., -3., 5., 9., 0., -4])
    >>> y = ivy.top_k(x, 3)
    >>> print(y)
    top_k(values=ivy.array([9., 5., 2.]), indices=ivy.array([4, 3, 0]))

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, 2, -4]), b=ivy.array([4., 5., 0.]))
    >>> y = ivy.top_k(2)
    >>> print(y)
    {
        a: [
            values = ivy.array([ 2, -1]),
            indices = ivy.array([1, 0])
        ],
        b: [
            values = ivy.array([5., 4.]),
            indices = ivy.array([1, 0])
        ]
    }
    """
    return current_backend(x).top_k(
        x, k, axis=axis, largest=largest, sorted=sorted, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def fliplr(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Flip array in the left/right direction. Flip the entries in each column in the
    left/right direction. Columns are preserved, but appear in a different order than
    before.

    Parameters
    ----------
    m
        The array to be flipped. Must be at least 2-D.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array corresponding to input array with elements
        order reversed along axis 1.

    Examples
    --------
    >>> m = ivy.diag([1, 2, 3])
    >>> ivy.fliplr(m)
    ivy.array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    """
    return ivy.current_backend().fliplr(m, copy=copy, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def i0(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the Bessel i0 function of x element-wise.

    Parameters
    ----------
    x
        Array input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with the modified Bessel function
        evaluated at each of the elements of x.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> ivy.i0(x)
    ivy.array([1.26606588, 2.2795853 , 4.88079259])
    """
    return ivy.current_backend(x).i0(x, out=out)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


def _set_pad_area(padded, axis, width_pair, value_pair):
    if width_pair[0] > 0:
        left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
        padded[left_slice] = value_pair[0]
    if width_pair[1] > 0:
        right_slice = _slice_at_axis(
            slice(padded.shape[axis] - width_pair[1], None), axis
        )
        padded[right_slice] = value_pair[1]
    return padded


def _get_edges(padded, axis, width_pair):
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]
    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]
    return left_edge, right_edge


def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    edge_pair = _get_edges(padded, axis, width_pair)
    if width_pair[0] > 0:
        left_ramp = ivy.linspace(
            end_value_pair[0],
            ivy.array(edge_pair[0].squeeze(axis)),
            num=width_pair[0],
            endpoint=False,
            dtype=ivy.Dtype(str(padded.dtype)),
            axis=axis,
        )
    else:
        left_ramp = ivy.empty((0,))
    if width_pair[1] > 0:
        right_ramp = ivy.flip(
            ivy.linspace(
                end_value_pair[1],
                ivy.array(edge_pair[1].squeeze(axis)),
                num=width_pair[1],
                endpoint=False,
                dtype=ivy.Dtype(str(padded.dtype)),
                axis=axis,
            ),
            axis=axis,
        )
    else:
        right_ramp = ivy.empty((0,))
    return left_ramp, right_ramp


def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    max_length = right_index - left_index
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length
    left_slice = _slice_at_axis(slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    left_stat = ivy.round(left_stat) if "int" in left_chunk.dtype else left_stat
    if left_length == right_length == max_length:
        return left_stat, left_stat
    right_slice = _slice_at_axis(slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    right_stat = ivy.round(right_stat) if "int" in right_chunk.dtype else right_stat
    return left_stat, right_stat


def _set_reflect_both(padded, axis, width_pair, method, include_edge=False):
    left_pad, right_pad = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad
    if include_edge:
        edge_offset = 1
    else:
        edge_offset = 0
        old_length -= 1
    if left_pad > 0:
        chunk_length = min(old_length, left_pad)
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]
        if method == "odd":
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        left_pad -= chunk_length
    if right_pad > 0:
        chunk_length = min(old_length, right_pad)
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]
        if method == "odd":
            edge_slice = _slice_at_axis(slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        right_pad -= chunk_length
    return left_pad, right_pad, padded


def _set_wrap_both(padded, axis, width_pair):
    left_pad, right_pad = width_pair
    period = padded.shape[axis] - right_pad - left_pad
    new_left_pad = 0
    new_right_pad = 0
    if left_pad > 0:
        right_slice = _slice_at_axis(
            slice(
                -right_pad - min(period, left_pad),
                -right_pad if right_pad != 0 else None,
            ),
            axis,
        )
        right_chunk = padded[right_slice]
        if left_pad > period:
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk
    if right_pad > 0:
        left_slice = _slice_at_axis(
            slice(
                left_pad,
                left_pad + min(period, right_pad),
            ),
            axis,
        )
        left_chunk = padded[left_slice]
        if right_pad > period:
            pad_area = _slice_at_axis(slice(-right_pad, -right_pad + period), axis)
            new_right_pad = right_pad - period
        else:
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk
    return new_left_pad, new_right_pad, padded


def _pad_simple(array, pad_width, fill_value=None):
    new_shape = tuple(
        left + size + right for size, (left, right) in zip(array.shape, pad_width)
    )
    padded = ivy.zeros(new_shape, dtype=array.dtype)
    if fill_value is not None:
        padded = ivy.ones_like(padded) * fill_value
    original_area_slice = tuple(
        slice(left, left + size) for size, (left, right) in zip(array.shape, pad_width)
    )
    padded[original_area_slice] = array
    return padded, original_area_slice


def _to_pairs(x, n):
    if ivy.isscalar(x):
        return ((x, x),) * n
    elif len(x) == 2 and ivy.isscalar(x[0]):
        return ((x[0], x[1]),) * n
    elif len(x) != n:
        ivy.utils.assertions.check_equal(
            ivy.asarray(list(x)).shape,
            (n, 2),
            message=(
                "tuple argument should contain "
                "ndim pairs where ndim is the number of "
                "the input's dimensions"
            ),
            as_array=False,
        )
    return x


def _to_dilated(x, n):
    if ivy.isscalar(x):
        return ((x, x, x),) * n
    elif len(x) == 3 and ivy.isscalar(x[0]):
        return ((x[0], x[1], x[2]),) * n
    elif len(x) != n:
        ivy.utils.assertions.check_equal(
            ivy.asarray(list(x)).shape,
            (n, 3),
            message=(
                "tuple argument should contain "
                "ndim groups where ndim is the number of "
                "the input's dimensions"
            ),
            as_array=False,
        )
    return x


def _check_tuple_arg(arg, name, force_integer=True):
    is_scalar = ivy.isscalar if not force_integer else ivy.is_int_dtype
    flag_assert = False
    if isinstance(arg, (tuple, list)):
        for nested in arg:
            if isinstance(nested, (tuple, list)):
                for sub_nested in nested:
                    if not is_scalar(sub_nested):
                        flag_assert = True
                        break
            elif not is_scalar(nested):
                flag_assert = True
    elif not is_scalar(arg):
        flag_assert = True
    if flag_assert:
        if not force_integer:
            raise ivy.utils.exceptions.IvyException(
                name + " should be scalar, tuple of scalars or tuple of scalar tuples"
            )
        else:
            raise ivy.utils.exceptions.IvyException(
                name + " should be int, tuple of ints or tuple of int tuples"
            )


def _check_arguments(
    mode,
    pad_width,
    stat_length,
    constant_values,
    end_values,
    reflect_type,
):
    ivy.utils.assertions.check_true(
        callable(mode)
        or mode
        in [
            "constant",
            "dilated",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
        message="the provided mode is not supported",
    )
    _check_tuple_arg(pad_width, "pad_width")
    if mode not in ["dilated"]:
        ivy.utils.assertions.check_true(
            all(element[1] >= 0 for element in ivy.ndenumerate(pad_width)),
            message="the pad_widths must be greater or equal to zero",
        )
    if mode in ["maximum", "mean", "median", "minimum"]:
        _check_tuple_arg(stat_length, "stat_length")
        ivy.utils.assertions.check_true(
            all(element[1] > 0 for element in ivy.ndenumerate(stat_length)),
            message="the stat lengths must be greater than zero",
        )
    elif mode == "constant":
        _check_tuple_arg(constant_values, "constant_values", force_integer=False)
    elif mode == "linear_ramp":
        _check_tuple_arg(end_values, "end_values", force_integer=False)
    ivy.utils.assertions.check_true(
        reflect_type in ["even", "odd"],
        message="the provided reflect_type is not supported",
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def pad(
    input: Union[ivy.Array, ivy.NativeArray],
    pad_width: Union[Iterable[Tuple[int]], int],
    /,
    *,
    mode: Union[
        Literal[
            "constant",
            "dilated",
            "edge",
            "linear_ramp",
            "maximum",
            "mean",
            "median",
            "minimum",
            "reflect",
            "symmetric",
            "wrap",
            "empty",
        ],
        Callable,
    ] = "constant",
    stat_length: Union[Iterable[Tuple[int]], int] = 1,
    constant_values: Union[Iterable[Tuple[Number]], Number] = 0,
    end_values: Union[Iterable[Tuple[Number]], Number] = 0,
    reflect_type: Literal["even", "odd"] = "even",
    **kwargs: Optional[Any],
) -> ivy.Array:
    """
    Pad an array.

    Parameters
    ----------
    input
        Input array to pad.
    pad_width
        Number of values padded to the edges of each axis.
             - ((before_1, after_1), … (before_N, after_N)) yields unique pad widths
               for each axis.
             - ((before, after),) yields same before and after pad for each axis.
             - pad (integer) is shortcut for before = after = pad width for all axes.
    mode
        One of the following string values or a user-supplied function.
             - "constant": Pads with a constant value.
             - "edge": Pads with the input's edge values.
             - "linear_ramp": Pads with the linear ramp between end_value
               and the input's edge value.
             - "maximum": Pads with the maximum value of all or part of the vector
               along each axis.
             - "mean": Pads with the mean value of all or part of the vector along
               each axis.
             - "median": Pads with the median value of all or part of the vector
               along each axis.
             - "minimum": Pads with the minimum value of all or part of the vector
               along each axis.
             - "reflect": Pads with the reflection mirrored on the first and last
               values of the vector along each axis.
             - "symmetric": Pads with the reflection of the vector mirrored along
               the edge of the input.
             - "wrap": Pads with the wrap of the vector along the axis.
               The first values are used to pad the end and the end values are used
               to pad the beginning.
             - "empty": Pads with undefined values.
             - <function>: Pads with a user-defined padding function. The padding
               function should modify a rank 1 array following the signature
               `padding_func(vector, iaxis_pad_width, iaxis, kwargs)`, where:
                    - `vector` is a rank 1 array already padded with zeros. Padded
                      values are `vector[:iaxis_pad_width[0]]` and
                      `vector[-iaxis_pad_width[1]:]`.
                    - `iaxis_pad_width` is a 2-tuple of ints, where
                      `iaxis_pad_width[0]` represents the number of values padded at
                      the beginning of `vector` and `iaxis_pad_width[1]` represents
                      the number of values padded at the end of `vector`.
                    - `iaxis` is the axis currently being calculated.
                    - `kwargs` is a dict of keyword arguments the function requires.
    stat_length
        Used in "maximum", "mean", "median", and "minimum". Number of values at edge
        of each axis used to calculate the statistic value.
             - ((before_1, after_1), … (before_N, after_N)) yields unique statistic
               lengths for each axis.
             - ((before, after),) yields same before and after statistic lengths for
               each axis.
             - stat_length (integer) is a shortcut for before = after = stat_length
               length for all axes.
             - None uses the entire axis.
    constant_values
        Used in "constant". The values to set the padded values for each axis.
             - ((before_1, after_1), ... (before_N, after_N)) yields unique pad
               constants for each axis.
             - ((before, after),) yields same before and after constants for each axis.
             - constant (integer) is a shortcut for before = after = constant for
               all axes.
    end_values
        Used in "linear_ramp". The values used for the ending value of the linear_ramp
        and that will form the edge of the padded array.
             - ((before_1, after_1), ... (before_N, after_N)) yields unique end values
               for each axis.
             - ((before, after),) yields same before and after end values for each axis
             - end (integer) is a shortcut for before = after = end for all axes.
    reflect_type
        Used in "reflect", and "symmetric". The "even" style is the default with an
        unaltered reflection around the edge value. For the "odd" style, the extended
        part of the array is created by subtracting the reflected values from two
        times the edge value.

    Returns
    -------
    ret
        Padded array of the same rank as the input but with shape increased according
        to pad_width.


    Both the description and the type hints above assume an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> padding = ((1, 1), (2, 2))
    >>> y = ivy.pad(x, padding, mode="constant", constant_values=0)
    >>> print(y)
    ivy.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 2, 3, 0, 0],
               [0, 0, 4, 5, 6, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> padding = ((1, 1), (2, 2))
    >>> y = ivy.pad(x, padding, mode="reflect")
    >>> print(y)
    ivy.array([[6, 5, 4, 5, 6, 5, 4],
               [3, 2, 1, 2, 3, 2, 1],
               [6, 5, 4, 5, 6, 5, 4],
               [3, 2, 1, 2, 3, 2, 1]])

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]])
    >>> padding = ((1, 1), (2, 2))
    >>> y = ivy.pad(x, padding, mode="symmetric")
    >>> print(y)
    ivy.array([[2, 1, 1, 2, 3, 3, 2],
               [2, 1, 1, 2, 3, 3, 2],
               [5, 4, 4, 5, 6, 6, 5],
               [5, 4, 4, 5, 6, 6, 5]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[1, 2, 3], [4, 5, 6]])
    >>> padding = ((1, 1), (2, 2))
    >>> y = ivy.pad(x, padding, mode="constant", constant_values=7)
    >>> print(y)
    ivy.array([[7, 7, 7, 7, 7, 7, 7],
               [7, 7, 1, 2, 3, 7, 7],
               [7, 7, 4, 5, 6, 7, 7],
               [7, 7, 7, 7, 7, 7, 7]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([4, 5, 6]))
    >>> padding = (1, 1)
    >>> y = ivy.pad(x, padding, mode="constant")
    >>> print(y)
    {
        a: ivy.array([0, 0, 1, 2, 0]),
        b: ivy.array([0, 4, 5, 6, 0])
    }
    """
    _check_arguments(
        mode,
        pad_width,
        stat_length,
        constant_values,
        end_values,
        reflect_type,
    )
    if mode == "dilated":
        pad_width = _to_dilated(pad_width, input.ndim)
        if ivy.as_ivy_dtype(type(constant_values)) != input.dtype:
            padding_value = ivy.native_array(constant_values, dtype=input.dtype)
        else:
            padding_value = constant_values
        padded = _interior_pad(input, padding_value, pad_width)
        return padded
    pad_width = _to_pairs(pad_width, len(input.shape))
    if callable(mode):
        func = mode
        padded, _ = _pad_simple(input, pad_width, fill_value=0)
        for axis in range(padded.ndim):
            padded = ivy.moveaxis(padded, axis, -1)
            inds = ivy.ndindex(padded.shape[:-1])
            for ind in inds:
                padded[ind] = func(padded[ind], pad_width[axis], axis, kwargs)
        return padded
    padded, original_area_slice = _pad_simple(input, pad_width)
    axes = range(padded.ndim)
    stat_functions = {
        "maximum": ivy.max,
        "minimum": ivy.min,
        "mean": ivy.mean,
        "median": ivy.median,
    }
    if mode == "constant":
        constant_values = _to_pairs(constant_values, padded.ndim)
        constant_values = tuple(tuple(map(ivy.array, pair)) for pair in constant_values)
        for axis, width_pair, value_pair in zip(axes, pad_width, constant_values):
            padded = _set_pad_area(padded, axis, width_pair, value_pair)
    elif mode == "empty":
        pass
    elif mode == "edge":
        for axis, width_pair in zip(axes, pad_width):
            edge_pair = _get_edges(padded, axis, width_pair)
            padded = _set_pad_area(padded, axis, width_pair, edge_pair)
    elif mode == "linear_ramp":
        end_values = _to_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            ramp_pair = _get_linear_ramps(padded, axis, width_pair, value_pair)
            padded = _set_pad_area(padded, axis, width_pair, ramp_pair)
    elif mode in stat_functions:
        func = stat_functions[mode]
        stat_length = _to_pairs(stat_length, padded.ndim)
        if mode == "median":
            ivy.utils.assertions.check_true(
                ivy.is_float_dtype(input),
                message="median interpolation is only supported for floats",
            )
        for axis, width_pair, length_pair in zip(axes, pad_width, stat_length):
            stat_pair = _get_stats(padded, axis, width_pair, length_pair, func)
            padded = _set_pad_area(padded, axis, width_pair, stat_pair)
    elif mode in {"reflect", "symmetric"}:
        include_edge = True if mode == "symmetric" else False
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if input.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                padded = _set_pad_area(
                    padded, axis, (left_index, right_index), edge_pair
                )
                continue
            while left_index > 0 or right_index > 0:
                left_index, right_index, padded = _set_reflect_both(
                    padded, axis, (left_index, right_index), reflect_type, include_edge
                )
    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width):
            while left_index > 0 or right_index > 0:
                left_index, right_index, padded = _set_wrap_both(
                    padded, axis, (left_index, right_index)
                )
    return padded


pad.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def vsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Sequence[int], ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Split an array vertically into multiple sub-arrays.

    Parameters
    ----------
    ary
        Array input.
    indices_or_sections
        If indices_or_sections is an integer n, the array is split into n
        equal sections, provided that n must be a divisor of the split axis.
        If indices_or_sections is a sequence of ints or 1-D array,
        then input is split at each of the indices.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        input array split vertically.

    Examples
    --------
    >>> ary = ivy.array(
        [[[0.,  1.],
          [2.,  3.]],
         [[4.,  5.],
          [6.,  7.]]]
        )
    >>> ivy.vsplit(ary, 2)
    [ivy.array([[[0., 1.], [2., 3.]]]), ivy.array([[[4., 5.], [6., 7.]]])])
    """
    return ivy.current_backend(ary).vsplit(ary, indices_or_sections, copy=copy)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_device_shifting
def dsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Sequence[int], ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Split an array into multiple sub-arrays along the 3rd axis.

    Parameters
    ----------
    ary
        Array input.
    indices_or_sections
        If indices_or_sections is an integer n, the array is split into n sections.
        If the array is divisible by n along the 3rd axis, each section will be of
        equal size. If input is not divisible by n, the sizes of the first
        int(ary.size(0) % n) sections will have size int(ary.size(0) / n) + 1, and
        the rest will have size int(ary.size(0) / n).
        If indices_or_sections is a sequence of ints or 1-D array,
        then input is split at each of the indices.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        input array split along the 3rd axis.

    Examples
    --------
    >>> ary = ivy.array(
        [[[ 0.,   1.,   2.,   3.],
          [ 4.,   5.,   6.,   7.]],
         [[ 8.,   9.,  10.,  11.],
          [12.,  13.,  14.,  15.]]]
        )
    >>> ivy.dsplit(ary, 2)
    [ivy.array([[[ 0.,  1.], [ 4.,  5.]], [[ 8.,  9.], [12., 13.]]]),
     ivy.array([[[ 2.,  3.], [ 6.,  7.]], [[10., 11.], [14., 15.]]])]
    """
    return ivy.current_backend(ary).dsplit(ary, indices_or_sections, copy=copy)


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_device_shifting
def atleast_1d(
    *arys: Union[ivy.Array, ivy.NativeArray, bool, Number],
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Convert inputs to arrays with at least one dimension. Scalar inputs are converted to
    1-dimensional arrays, whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more input arrays.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        An array, or list of arrays, each with atleast 1D.
        Copies are made only if necessary.

    Examples
    --------
    >>> ary1 = ivy.array(5)
    >>> ivy.atleast_1d(ary1)
    ivy.array([5])
    >>> ary2 = ivy.array([[3,4]])
    >>> ivy.atleast_1d(ary2)
    ivy.array([[3, 4]])
    >>> ivy.atleast_1d(6,7,8)
    [ivy.array([6]), ivy.array([7]), ivy.array([8])]
    """
    return ivy.current_backend().atleast_1d(*arys, copy=copy)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def dstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Stack arrays in sequence depth wise (along third axis).

    Parameters
    ----------
    arrays
        Sequence of arrays to be stacked.

    Returns
    -------
    ret
        The array formed by stacking the given arrays.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([2, 3, 4])
    >>> ivy.dstack((x, y))
    ivy.array([[[1, 2],
                [2, 3],
                [3, 4]]])
    >>> x = ivy.array([[1], [2], [3]])
    >>> y = ivy.array([[2], [3], [4]])
    >>> ivy.dstack((x, y))
    ivy.array([[[1, 2]],
               [[2, 3]],
               [[3, 4]]])
    """
    return ivy.current_backend().dstack(arrays, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_device_shifting
def atleast_2d(
    *arys: Union[ivy.Array, ivy.NativeArray],
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Convert inputs to arrays with at least two dimension. Scalar inputs are converted to
    2-dimensional arrays, whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more array-like sequences. Non-array inputs are
        converted to arrays. Arrays that already have two or more
        dimensions are preserved.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        An array, or list of arrays, each with atleast 2D.
        Copies are made only if necessary.

    Examples
    --------
    >>> ary1 = ivy.array(5)
    >>> ivy.atleast_2d(ary1)
    ivy.array([[5]])
    >>> ary2 = ivy.array([[[3,4]]])
    >>> ivy.atleast_2d(ary2)
    ivy.array([[[3, 4]]])
    >>> ivy.atleast_2d(6,7,8)
    [ivy.array([[6]]), ivy.array([[7]]), ivy.array([[8]])]
    """
    return ivy.current_backend().atleast_2d(*arys, copy=copy)


@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_device_shifting
def atleast_3d(
    *arys: Union[ivy.Array, ivy.NativeArray, bool, Number],
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Convert inputs to arrays with at least three dimension. Scalar inputs are converted
    to 3-dimensional arrays, whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more array-like sequences. Non-array inputs are
        converted to arrays. Arrays that already have three or more
        dimensions are preserved.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        An array, or list of arrays, each with a.ndim >= 3. Copies
        are avoided where possible, and views with three or more
        dimensions are returned. For example, a 1-D array of shape
        (N,) becomes a view of shape (1, N, 1), and a 2-D array of
        shape (M, N) becomes a view of shape (M, N, 1).

    Examples
    --------
    >>> ary1 = ivy.array([5,6])
    >>> ivy.atleast_3d(ary1)
    ivy.array([[[5],
            [6]]])
    >>> ary2 = ivy.array([[[3,4]]])
    >>> ivy.atleast_3d(ary2)
    ivy.array([[[3, 4]]])
    >>> ary3 = ivy.array([[3,4],[9,10]])
    >>> ivy.atleast_3d(6,7,ary3)
    [ivy.array([[[6]]]), ivy.array([[[7]]]), ivy.array([[[ 3],
            [ 4]],

           [[ 9],
            [10]]])]
    """
    return ivy.current_backend().atleast_3d(*arys, copy=copy)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def take_along_axis(
    arr: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Take values from the input array by matching 1d index and data slices.

    Parameters
    ----------
    arr
        The source array.
    indices
        The indices of the values to extract.
    axis
        The axis over which to select values.
        If axis is None, arr is treated as a flattened 1D array.
    mode
        One of: 'clip', 'fill', 'drop'. Parameter controlling how out-of-bounds indices
        will be handled.
    out
        The output array.

    Returns
    -------
    ret
        The returned array has the same shape as `indices`.

    Examples
    --------
    >>> arr = ivy.array([[4, 3, 5], [1, 2, 1]])
    >>> indices = ivy.array([[0, 1, 1], [2, 0, 0]])
    >>> y = ivy.take_along_axis(arr, indices, 1)
    >>> print(y)
    ivy.array([[4, 3, 3], [1, 1, 1]])
    """
    return ivy.current_backend(arr).take_along_axis(
        arr, indices, axis, mode=mode, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_view
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def hsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Sequence[int], ivy.Array, ivy.NativeArray],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[ivy.Array]:
    """
    Split an array into multiple sub-arrays horizontally.

    Parameters
    ----------
    ary
        Array input.
    indices_or_sections
        If indices_or_sections is an integer n, the array is split into n
        equal sections, provided that n must be a divisor of the split axis.
        If indices_or_sections is a tuple of ints, then input is split at each of
        the indices in the tuple.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.

    Returns
    -------
    ret
        input array split horizontally.

    Examples
    --------
    >>> ary = ivy.array(
            [[0.,  1., 2., 3.],
             [4.,  5., 6,  7.],
             [8.,  9., 10., 11.],
             [12., 13., 14., 15.]]
            )
    >>> ivy.hsplit(ary, 2)
        [ivy.array([[ 0.,  1.],
                    [ 4.,  5.],
                    [ 8.,  9.],
                    [12., 13.]]),
         ivy.array([[ 2.,  3.],
                    [ 6.,  7.],
                    [10., 11.],
                    [14., 15.]])]
    """
    return ivy.current_backend(ary).hsplit(ary, indices_or_sections, copy=copy)


@handle_exceptions
@inputs_to_native_shapes
def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    """
    Broadcasts shapes.

    Parameters
    ----------
    shapes
        The shapes to broadcast.

    Returns
    -------
    ret
        The broadcasted shape.

    Examples
    --------
    >>> x = [(3, 3), (3, 1)]
    >>> print(ivy.broadcast_shapes(x))
    (3, 3)

    >>> print(ivy.broadcast_shapes([(3, 3),(3, 1),(1, 3)]))
    (3, 3)
    """
    return ivy.current_backend().broadcast_shapes(*shapes)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_view
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_device_shifting
def expand(
    x: Union[ivy.Array, ivy.NativeArray],
    shape: Union[ivy.Shape, ivy.NativeShape],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Broadcast the input Array following the given shape and the broadcast rule.

    Parameters
    ----------
    x
        Array input.
    shape
        A 1-D Array indicates the shape you want to expand to,
        following the broadcast rule.
    copy
        boolean indicating whether to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy and must
        raise a ValueError in case a copy would be necessary.
        If None, the function must reuse existing memory buffer if possible
        and copy otherwise. Default: ``None``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Output Array
    """
    return ivy.current_backend(x).expand(x, shape, out=out, copy=copy)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
def put_along_axis(
    arr: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    values: Union[ivy.Array, ivy.NativeArray],
    axis: int,
    /,
    *,
    mode: str = "raise",
    out: Optional[ivy.Array] = None,
) -> None:
    """
    Put values into the input array by matching 1d index and data slices along a
    specified axis.

    Parameters
    ----------
    arr : array_like
        The input array to modify.
    indices : array_like
        The indices of the values to put into `arr`.
    values : array_like
        The values to put into `arr`.
    axis : int
        The axis over which to put the `values`.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will be handled.
        The following modes are available:

        - 'raise': a `ValueError` is raised when an index is out of bounds.
        - 'wrap': the index is wrapped around to the corresponding index
        at the other end of the axis.
        - 'clip': the index is clipped to the closest in-bounds index.
    out : ndarray, optional
        Output array in which to place the result.
        If not specified, a new array is created.

    Returns
    -------
    None

    Examples
    --------
    >>> arr = ivy.array([[4, 3, 5], [1, 2, 1]])
    >>> indices = ivy.array([[0, 1, 1], [2, 0, 0]])
    >>> values = ivy.array([[9, 8, 7], [6, 5, 4]])
    >>> put_along_axis(arr, indices, values, 1, mode='clip')
    >>> print(arr)
    ivy.array([[3, 7, 5],
               [6, 4, 1]])
    """
    if out is None:
        out = ivy.zeros_like(arr)

    indices = ivy.expand_dims(indices, axis=axis)
    values = ivy.expand_dims(values, axis=axis)

    stacked = ivy.concat((arr, values), axis=axis)

    sorted_indices = ivy.argsort(indices, axis=axis)
    sorted_stacked = ivy.take_along_axis(stacked, sorted_indices, axis=axis)

    arr = ivy.where(
        ivy.expand_dims(sorted_indices < arr.shape[axis], axis=axis),
        sorted_stacked,
        arr,
    )

    if mode == "clip":
        indices = ivy.clip(indices, 0, arr.shape[axis] - 1)
    elif mode == "wrap":
        indices = ivy.mod(indices, arr.shape[axis])

    arr = ivy.where(
        ivy.expand_dims(sorted_indices < arr.shape[axis], axis=axis), arr, values
    )

    ivy.assign(out, arr)


def _check_bounds(shape0, shape1, strides1, itemsize):
    numel0 = math.prod(shape0)
    ndim1 = len(shape1)
    return (
        sum((shape1[i] - 1) * strides1[i] for i in range(ndim1)) + itemsize
        <= numel0 * itemsize
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_shapes
@inputs_to_ivy_arrays
def as_strided(
    x: Union[ivy.Array, ivy.NativeArray],
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    strides: Sequence[int],
    /,
) -> ivy.Array:
    """
    Create a copy of the input array with the given shape and strides.

    Parameters
    ----------
    x
        Input Array.
    shape
        The shape of the new array.
    strides
        The strides of the new array (specified in bytes).

    Returns
    -------
    ret
        Output Array

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> ivy.as_strided(x, (4, 3), (8, 8))
    ivy.array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]])
    """
    itemsize = x.itemsize
    if not _check_bounds(x.shape, shape, strides, itemsize):
        raise ivy.exceptions.IvyException("attempted unsafe memory access")
    if any(strides[i] % itemsize != 0 for i in range(len(strides))):
        raise ivy.exceptions.IvyException("strides must be multiple of itemsize")

    src = memoryview(ivy.to_numpy(x)).cast("b")

    src_ind = ivy.inner(
        ivy.indices(shape).reshape((len(shape), -1)).T,
        ivy.array(strides),
    )
    src_ind = ivy.expand_dims(src_ind, axis=-1)
    src_ind = src_ind + ivy.arange(itemsize)
    src_ind = ivy.reshape(src_ind, (-1,)).to_numpy()

    temp_list = [src[i] for i in src_ind]
    temp_array = ivy.asarray(temp_list, dtype=ivy.int8)
    result = bytearray(temp_array.to_numpy())

    return ivy.reshape(
        ivy.frombuffer(result, dtype=x.dtype, count=math.prod(shape)),
        shape,
    )


as_strided.unsupported_dtypes = ("bfloat16",)
as_strided.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def concat_from_sequence(
    input_sequence: Union[
        Tuple[Union[ivy.Array, ivy.NativeArray]],
        List[Union[ivy.Array, ivy.NativeArray]],
    ],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Concatenate a sequence of arrays along a new or an existing axis.

    Parameters
    ----------
    input_sequence
        A sequence of arrays.
    new_axis
        Insert and concatenate on a new axis or not,
        default 0 means do not insert new axis.
        new_axis = 0: concatenate
        new_axis = 1: stack
    axis
        axis along which the arrays will be concatenated.

    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Output Array
    """
    return current_backend(input_sequence).concat_from_sequence(
        input_sequence, new_axis=new_axis, axis=axis, out=out
    )


def _slice(operand, start_indices, limit_indices, strides=None):
    strides = [1] * len(operand.shape) if strides is None else strides

    full_slice = ()
    for i, _ in enumerate(operand.shape):
        strides_i = int(strides[i])
        start_i = int(start_indices[i])
        limit_i = int(limit_indices[i])
        full_slice += (slice(start_i, limit_i, strides_i),)
    return operand[full_slice]


def _slice_along_axis(x, start=0, stop=None, stride=1, axis=0):
    if axis >= 0:
        slices = [slice(None)] * axis + [slice(start, stop, stride)]
    else:
        slices = [Ellipsis, slice(start, stop, stride)] + [slice(None)] * (-1 - axis)
    return x[tuple(slices)]


def _interior_pad(operand, padding_value, padding_config):
    for axis, (_, _, interior) in enumerate(padding_config):
        if interior > 0:
            new_shape = list(operand.shape)
            new_shape[axis] = new_shape[axis] + (new_shape[axis] - 1) * interior
            new_array = ivy.full(new_shape, padding_value)
            src_indices = ivy.arange(operand.shape[axis])
            dst_indices = src_indices * (interior + 1)
            index_tuple = [slice(None)] * operand.ndim
            index_tuple[axis] = dst_indices
            new_array[tuple(index_tuple)] = operand
            operand = new_array

    start_indices = [0] * operand.ndim
    limit_indices = [0] * operand.ndim
    for axis, (low, high, _) in enumerate(padding_config):
        if low < 0:
            start_indices[axis] = abs(low)
        if high < 0:
            limit_indices[axis] = high
        else:
            limit_indices[axis] = operand.shape[axis] + 1
    padded = _slice(operand, start_indices, limit_indices)

    pad_width = [(0, 0)] * operand.ndim
    for axis, (low, high, _) in enumerate(padding_config):
        if low > 0 and high > 0:
            pad_width[axis] = (low, high)
        elif low > 0 and not high > 0:
            pad_width[axis] = (low, 0)
        elif high > 0 and not low > 0:
            pad_width[axis] = (0, high)
    padded = ivy.constant_pad(padded, pad_width, value=padding_value)
    return padded


def _interleave(a, b, axis):
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
    b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
    a = _interior_pad(a, 0.0, a_pad)
    b = _interior_pad(b, 0.0, b_pad)
    return ivy.add(a, b)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def associative_scan(
    x: Union[ivy.Array, ivy.NativeArray],
    fn: Callable,
    /,
    *,
    reverse: bool = False,
    axis: int = 0,
) -> ivy.Array:
    """
    Perform an associative scan over the given array.

    Parameters
    ----------
    x
        The array to scan over.
    fn
        The associative function to apply.
    reverse
        Whether to scan in reverse with respect to the given axis.
    axis
        The axis to scan over.

    Returns
    -------
    ret
        The result of the scan.
    """
    elems = [x]

    if reverse:
        elems = [ivy.flip(elem, axis=[axis]) for elem in elems]

    def _combine(a, b):
        a = a[0]
        b = b[0]
        if a.shape[axis] == 0:
            return [a]
        c = fn(a, b)
        return [c]

    def _scan(elems):
        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [_slice_along_axis(elem, 0, -1, stride=2, axis=axis) for elem in elems],
            [_slice_along_axis(elem, 1, None, stride=2, axis=axis) for elem in elems],
        )

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = _combine(
                [_slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [_slice_along_axis(e, 2, None, stride=2, axis=axis) for e in elems],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [_slice_along_axis(e, 2, None, stride=2, axis=axis) for e in elems],
            )
        even_elems = [
            ivy.concat([_slice_along_axis(elem, 0, 1, axis=axis), result], axis=axis)
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems)

    if reverse:
        scans = [ivy.flip(scanned, axis=[axis]) for scanned in scans]

    return ivy.reshape(ivy.asarray(scans), elems[0].shape)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def unique_consecutive(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[
    Union[ivy.Array, ivy.NativeArray],
    Union[ivy.Array, ivy.NativeArray],
    Union[ivy.Array, ivy.NativeArray],
]:
    """Eliminates all but the first element from every consecutive group of equivalent
    elements in ``x``.

    Parameters
    ----------
    x
        input array.

    axis
        the axis to apply unique on. If None, unique is applied on flattened ``x``.

    Returns
    -------
    ret
        a namedtuple ``(output, inverse_indices, counts)`` whose
        - first element has the field name ``output`` and is an array
          containing ``x`` with its equivalent consecutive elements eliminated.
        - second element has the field name ``inverse_indices`` and is an
          array containing the indices of ``output`` that reconstruct ``x``.
        - third element has the field name ``counts`` and is an array
          containing the number of occurrences for each unique value or array in ``x``.


    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([1, 1, 2, 2, 3, 1, 1, 2])
    >>> ivy..unique_consecutive(x)
    Results(values=ivy.array([1, 2, 3, 1, 2]),
        inverse_indices=ivy.array([0, 0, 1, 1, 2, 3, 3, 4]),
        counts=ivy.array([2, 2, 1, 2, 1]))
    """
    return ivy.current_backend(x).unique_consecutive(x, axis=axis)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
def fill_diagonal(
    a: Union[ivy.Array, ivy.NativeArray],
    v: Union[int, float],
    /,
    *,
    wrap: bool = False,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Fill the main diagonal of the given array of any dimensionality..

    Parameters
    ----------
    a
        Array at least 2D.
    v
        The value to write on the diagonal.
    wrap
        The diagonal ‘wrapped’ after N columns for tall matrices.

    Returns
    -------
    ret
        Array with the diagonal filled.
    """
    return ivy.current_backend(a).fill_diag(a, v, wrap=wrap)


# TODO add container and array methods
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def unfold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the mode-`mode` unfolding of `tensor` with modes starting at `0`.

    Parameters
    ----------
    input
        input tensor to be unfolded
    mode
        indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

    Returns
    -------
    ret
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return ivy.reshape(ivy.moveaxis(input, mode, 0), (input.shape[mode], -1), out=out)


# TODO add container and array methods
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def fold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: int,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Refolds the mode-`mode` unfolding into a tensor of shape `shape` In other words,
    refolds the n-mode unfolded tensor into the original tensor of the specified shape.

    Parameters
    ----------
    input
        unfolded tensor of shape ``(shape[mode], -1)``
    mode
        the mode of the unfolding
    shape
        shape of the original tensor before unfolding

    Returns
    -------
    ret
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return ivy.moveaxis(ivy.reshape(input, full_shape), 0, mode, out=out)


# TODO add container and array methods
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_unfold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: Optional[int] = 0,
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    ravel_tensors: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
):
    """Partial unfolding of a tensor while ignoring the specified number
        of dimensions at the beginning and the end.
        For instance, if the first dimension of the tensor is the number
        of samples, to unfold each sample, set skip_begin=1.
        This would, for each i in ``range(tensor.shape[0])``, unfold ``tensor[i, ...]``.

    Parameters
    ----------
    input
        tensor of shape n_samples x n_1 x n_2 x ... x n_i
    mode
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end
    ravel_tensors
        if True, the unfolded tensors are also flattened

    Returns
    -------
    ret
        partially unfolded tensor
    """
    if ravel_tensors:
        new_shape = [-1]
    else:
        new_shape = [input.shape[mode + skip_begin], -1]

    if skip_begin:
        new_shape = [input.shape[i] for i in range(skip_begin)] + new_shape

    if skip_end:
        new_shape += [input.shape[-i] for i in range(1, 1 + skip_end)]

    return ivy.reshape(
        ivy.moveaxis(input, mode + skip_begin, skip_begin), new_shape, out=out
    )


# TODO add container and array methods
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_fold(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    mode: int,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    skip_begin: Optional[int] = 1,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Re-folds a partially unfolded tensor.

    Parameters
    ----------
    input
        a partially unfolded tensor
    mode
        indexing starts at 0, therefore mode is in range(0, tensor.ndim)
    shape
        the shape of the original full tensor (including skipped dimensions)
    skip_begin
        number of dimensions left untouched at the beginning

    Returns
    -------
    ret
        partially re-folded tensor
    """
    transposed_shape = list(shape)
    mode_dim = transposed_shape.pop(skip_begin + mode)
    transposed_shape.insert(skip_begin, mode_dim)
    return ivy.moveaxis(
        ivy.reshape(input, transposed_shape), skip_begin, skip_begin + mode, out=out
    )


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_tensor_to_vec(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    skip_begin: Optional[int] = 1,
    skip_end: Optional[int] = 0,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Partial vectorization of a tensor while ignoring the specified dimension at the
    beginning and the end.

    Parameters
    ----------
    input
        tensor to partially vectorise
    skip_begin
        number of dimensions to leave untouched at the beginning
    skip_end
        number of dimensions to leave untouched at the end

    Returns
    -------
    ret
        partially vectorised tensor with the
        `skip_begin` first and `skip_end` last dimensions untouched
    """
    return partial_unfold(
        input,
        mode=0,
        skip_begin=skip_begin,
        skip_end=skip_end,
        ravel_tensors=True,
        out=out,
    )


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def partial_vec_to_tensor(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
    skip_begin: Optional[int] = 1,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Refolds a partially vectorised tensor into a full one.

    Parameters
    ----------
    input
        a partially vectorised tensor
    shape
        the shape of the original full tensor (including skipped dimensions)
    skip_begin
        number of dimensions to leave untouched at the beginning
    Returns
    -------
    ret
        full tensor
    """
    return partial_fold(input, mode=0, shape=shape, skip_begin=skip_begin, out=out)


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def matricize(
    input: Union[ivy.Array, ivy.NativeArray],
    /,
    row_modes: Sequence[int],
    column_modes: Optional[Sequence[int]] = None,
    *,
    out: Optional[ivy.Array] = None,
):
    """
    Matricizes the given tensor.

    Parameters
    ----------
    input
        the input tensor
    row_modes
        modes to use as row of the matrix (in the desired order)
    column_modes
        modes to use as column of the matrix, in the desired order
        if None, the modes not in `row_modes` will be used in ascending order

    ret
    -------
    matrix : tensor of size (ivy.prod(input.shape[i] for i in row_modes), -1)
    """
    ndims = len(input.shape)
    row_indices = list(row_modes)

    if column_modes:
        column_indices = list(column_modes)
    else:
        column_indices = [i for i in range(ndims) if i not in row_indices]
        if sorted(column_indices + row_indices) != list(range(ndims)):
            msg = (
                "If you provide both column and row modes for the matricization then"
                " column_modes + row_modes must contain all the modes of the tensor."
                f" Yet, got row_modes={row_modes} and column_modes={column_modes}."
            )
            raise ValueError(msg)

    row_size, column_size = 1, 1
    row_size = int(ivy.prod([input.shape[i] for i in row_indices]))
    column_size = int(ivy.prod([input.shape[i] for i in column_indices]))

    return ivy.reshape(
        ivy.permute_dims(input, row_indices + column_indices),
        (row_size, column_size),
        out=out,
    )


@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def soft_thresholding(
    x: Union[ivy.Array, ivy.NativeArray],
    threshold: Union[float, ivy.Array, ivy.NativeArray],
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Soft-thresholding operator.

        sign(tensor) * max[abs(tensor) - threshold, 0]

    Parameters
    ----------
    x
      input array
    threshold
          float or ndarray with shape tensor.shape
        * If float the threshold is applied to the whole tensor
        * If ndarray, one threshold is applied per elements, 0 values are ignored

    Returns
    -------
    ivy.Array
        thresholded tensor on which the operator has been applied

    Examples
    --------
    Basic shrinkage

    >>> x = ivy.array([[1, -2, 1.5], [-4, 3, -0.5]])
    >>> soft_thresholding(x, 1.1)
    array([[ 0. , -0.9,  0.4],
           [-2.9,  1.9,  0. ]])


    Example with missing values

    >>> mask = ivy.array([[0, 0, 1], [1, 0, 1]])
    >>> soft_thresholding(x, mask*1.1)
    array([[ 1. , -2. ,  0.4],
           [-2.9,  3. ,  0. ]])

    See Also
    --------
    svd_thresholding : SVD-thresholding operator
    """
    # TODO x_max in clip should be None ideally
    # update when ivy.clip has been updated
    x_max = int(ivy.max(x))

    res = ivy.sign(x) * ivy.clip(ivy.abs(x) - threshold, 0, x_max)
    if ivy.exists(out):
        return ivy.inplace_update(out, res)
    return res
