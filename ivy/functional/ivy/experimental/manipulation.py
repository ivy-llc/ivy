from typing import (
    Optional,
    Union,
    Tuple,
    Iterable,
    Sequence,
    Generator,
    Callable,
    Any,
    Literal,
    List,
)
from numbers import Number
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions


@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def flatten(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Flattens input by reshaping it into a one-dimensional tensor.
        If start_dim or end_dim are passed, only dimensions starting
        with start_dim and ending with end_dim are flattened.
        The order of elements in input is unchanged.

    Parameters
    ----------
    x
        input array to flatten.
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
    if x.shape == ():
        x = ivy.reshape(x, (1, -1))[0, :]
    if start_dim == end_dim:
        return x
    if start_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
                {[-len(x.shape), len(x.shape) - 1]}, but got {start_dim}"
        )
    if end_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
                {[-len(x.shape), len(x.shape) - 1]}, but got {end_dim}"
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
    return ivy.reshape(x, tuple(lst), order=order)


flatten.mixed_function = True


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def moveaxis(
    a: Union[ivy.Array, ivy.NativeArray],
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Move axes of an array to new positions..

    Parameters
    ----------
    a
        The array whose axes should be reordered.
    source
        Original positions of the axes to move. These must be unique.
    destination
        Destination positions for each of the original axes.
        These must also be unique.
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
    return ivy.current_backend().moveaxis(a, source, destination, out=out)


@handle_exceptions
def ndenumerate(
    input: Iterable,
) -> Generator:
    """Multidimensional index iterator.

    Parameters
    ----------
    input
        Input array to iterate over.

    Returns
    -------
    ret
        An iterator yielding pairs of array coordinates and values.

    Examples
    --------
    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> for index, x in ivy.ndenumerate(a):
    >>>     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4
    """

    def _ndenumerate(input, t=None):
        if t is None:
            t = ()
        if ivy.is_ivy_array(input) and input.shape == ():
            input = ivy.to_scalar(input)
        if not hasattr(input, "__iter__"):
            yield t, input
        else:
            for i, v in enumerate(input):
                yield from _ndenumerate(v, t + (i,))

    return _ndenumerate(input)


@handle_exceptions
def ndindex(
    shape: Tuple,
) -> Generator:
    """Multidimensional index iterator.

    Parameters
    ----------
    shape
        The shape of the array to iterate over.

    Returns
    -------
    ret
        An iterator yielding array coordinates.

    Examples
    --------
    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> for index in ivy.ndindex(a):
    >>>     print(index)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)
    """

    def _iter_product(*args, repeat=1):
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

    args = []
    for s in range(len(shape)):
        args += [range(shape[s])]
    return _iter_product(*args)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def heaviside(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Heaviside step function for each element in x1.

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def flipud(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Flip array in the up/down direction.
    Flip the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m
        The array to be flipped.
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
    return ivy.current_backend().flipud(m, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def vstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Stack arrays in sequence vertically (row wise).

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def hstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> ivy.Array:
    """Stack arrays in sequence horizotally (column wise).

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def rot90(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.

    Parameters
    ----------
    m
        Input array of two or more dimensions.
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
    return ivy.current_backend(m).rot90(m, k=k, axes=axes, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def top_k(
    x: Union[ivy.Array, ivy.NativeArray],
    k: int,
    /,
    *,
    axis: Optional[int] = None,
    largest: Optional[bool] = True,
    out: Optional[tuple] = None,
) -> Tuple[ivy.Array, ivy.NativeArray]:
    """Returns the `k` largest elements of the given input array along a given axis.

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
    return current_backend(x).top_k(x, k, axis=axis, largest=largest, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def fliplr(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Flip array in the left/right direction.
    Flip the entries in each column in the left/right direction.
    Columns are preserved, but appear in a different order than before.

    Parameters
    ----------
    m
        The array to be flipped. Must be at least 2-D.
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
    return ivy.current_backend().fliplr(m, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def i0(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Bessel i0 function of x element-wise.

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
    return left_ramp.to_numpy(), right_ramp.to_numpy()


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
    left_chunk = ivy.array(padded[left_slice])
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True).astype(left_chunk.dtype)
    if left_length == right_length == max_length:
        return left_stat, left_stat
    right_slice = _slice_at_axis(slice(right_index - right_length, right_index), axis)
    right_chunk = ivy.array(padded[right_slice])
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True).astype(
        right_chunk.dtype
    )
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
    padded = padded.to_numpy()
    padded[original_area_slice] = array.to_numpy()
    return padded, original_area_slice


def _to_pairs(x, n):
    if ivy.isscalar(x):
        return ((x, x),) * n
    elif ivy.asarray(list(x)).shape == (2,):
        return ((x[0], x[1]),) * n
    else:
        ivy.assertions.check_equal(
            ivy.asarray(list(x)).shape,
            (n, 2),
            message="tuple argument should contain "
            "ndim pairs where ndim is the number of "
            "the input's dimensions",
        )
    return x


def _check_tuple_arg(arg, name):
    flag_assert = False
    if isinstance(arg, (tuple, list)):
        for nested in arg:
            if isinstance(nested, (tuple, list)):
                for sub_nested in nested:
                    if not isinstance(sub_nested, int):
                        flag_assert = True
                        break
            elif not isinstance(nested, int):
                flag_assert = True
    elif not isinstance(arg, int):
        flag_assert = True
    if flag_assert:
        raise ivy.exceptions.IvyException(
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
    ivy.assertions.check_true(
        callable(mode)
        or mode
        in [
            "constant",
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
    ivy.assertions.check_true(
        all(element[1] >= 0 for element in ivy.ndenumerate(pad_width)),
        message="the pad_widths must be greater or equal to zero",
    )
    if mode in ["maximum", "mean", "median", "minimum"]:
        if stat_length is None:
            raise ivy.exceptions.IvyException(
                "stat_length is required for mode: " + mode
            )
        else:
            _check_tuple_arg(stat_length, "stat_length")
            ivy.assertions.check_true(
                all(element[1] > 0 for element in ivy.ndenumerate(stat_length)),
                message="the stat lengths must be greater than zero",
            )
    elif mode == "constant":
        if constant_values is None:
            raise ivy.exceptions.IvyException(
                "constant_values is required for mode: " + mode
            )
        else:
            _check_tuple_arg(constant_values, "constant_values")
    elif mode == "linear_ramp":
        if end_values is None:
            raise ivy.exceptions.IvyException(
                "end_values is required for mode: " + mode
            )
        else:
            _check_tuple_arg(end_values, "end_values")
    ivy.assertions.check_true(
        reflect_type in ["even", "odd"],
        message="the provided reflect_type is not supported",
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def pad(
    input: Union[ivy.Array, ivy.NativeArray],
    pad_width: Union[Iterable[Tuple[int]], int],
    /,
    *,
    mode: Optional[
        Union[
            Literal[
                "constant",
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
        ]
    ] = "constant",
    stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
    constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = None,
    end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = None,
    reflect_type: Optional[Literal["even", "odd"]] = "even",
    **kwargs: Optional[Any],
) -> ivy.Array:
    """Pads an array.

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
    input = ivy.asarray(input, dtype=input.dtype)
    pad_width = _to_pairs(pad_width, input.ndim)
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
            ivy.assertions.check_true(
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
    padded = ivy.array(padded).to_native()
    return padded


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def vsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Tuple[int]],
    /,
) -> List[ivy.Array]:
    """Split an array into multiple sub-arrays along the 3rd axis.

    Parameters
    ----------
    ary
        Array input.
    indices_or_sections
        If indices_or_sections is an integer n, the array is split into n sections.
        If the array is divisible by n along the 3rd axis, each section will be of
        equal size. If input is not divisible by n, the sizes of the first
        int(ary.size(0) % n) sections will have size int(ary.size(0) / n) + 1,
        and the rest will have size int(ary.size(0) / n).
        If indices_or_sections is a tuple of ints, then input is split at each of
        the indices in the tuple.

    Returns
    -------
    ret
        input array split along the 3rd axis.

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
    return ivy.current_backend(ary).vsplit(ary, indices_or_sections)


@to_native_arrays_and_back
@handle_nestable
@handle_array_like_without_promotion
def dsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[ivy.Array]:
    """Split an array into multiple sub-arrays along the 3rd axis.

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
        If indices_or_sections is a tuple of ints, then input is split at each of
        the indices in the tuple.

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
    return ivy.current_backend(ary).dsplit(ary, indices_or_sections)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def atleast_1d(
    *arys: Union[ivy.Array, ivy.NativeArray, bool, Number],
) -> List[ivy.Array]:
    """Convert inputs to arrays with at least one dimension.
    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more input arrays.

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
    return ivy.current_backend().atleast_1d(*arys)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def dstack(
    arrays: Sequence[ivy.Array],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Stack arrays in sequence depth wise (along third axis).

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
    return ivy.current_backend().dstack(arrays)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def atleast_2d(
    *arys: Union[ivy.Array, ivy.NativeArray],
) -> List[ivy.Array]:
    """Convert inputs to arrays with at least two dimension.
    Scalar inputs are converted to 2-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more array-like sequences. Non-array inputs are
        converted to arrays. Arrays that already have two or more
        dimensions are preserved.

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
    return ivy.current_backend().atleast_2d(*arys)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def atleast_3d(
    *arys: Union[ivy.Array, ivy.NativeArray, bool, Number],
) -> List[ivy.Array]:
    """Convert inputs to arrays with at least three dimension.
    Scalar inputs are converted to 3-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys
        One or more array-like sequences. Non-array inputs are
        converted to arrays. Arrays that already have three or more
        dimensions are preserved.

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
    return ivy.current_backend().atleast_3d(*arys)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def take_along_axis(
    arr: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    axis: int,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Take values from the input array by matching 1d index and data slices.

    Parameters
    ----------
    arr
        The source array.
    indices
        The indices of the values to extract.
    axis
        The axis over which to select values.
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
    return ivy.current_backend(arr).take_along_axis(arr, indices, axis, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like_without_promotion
def hsplit(
    ary: Union[ivy.Array, ivy.NativeArray],
    indices_or_sections: Union[int, Tuple[int]],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Split an array into multiple sub-arrays horizontally.

    Parameters
    ----------
    ary
        Array input.
    indices_or_sections
        If indices_or_sections is an integer n, the array is split into n sections.
        If the array is divisible by n along the 3rd axis, each section will be of
        equal size. If input is not divisible by n, the sizes of the first
        int(ary.size(0) % n) sections will have size int(ary.size(0) / n) + 1,
        and the rest will have size int(ary.size(0) / n).
        If indices_or_sections is a tuple of ints, then input is split at each of
        the indices in the tuple.
    out
        optional output array, for writing the result to.

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
                    [14., 15.]]))
    """
    return ivy.current_backend(ary).hsplit(ary, indices_or_sections, out=out)


@handle_exceptions
def broadcast_shapes(shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    """Broadcasts shapes.

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
    return ivy.current_backend().broadcast_shapes(shapes)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_out_argument
@handle_array_like_without_promotion
def expand(
    x: Union[ivy.Array, ivy.NativeArray],
    shape: Union[ivy.Shape, ivy.NativeShape],
    /,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Broadcast the input Array following the given shape
    and the broadcast rule.

    Parameters
    ----------
    x
        Array input.
    shape
        A 1-D Array indicates the shape you want to expand to,
        following the broadcast rule
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Output Array
    """
    ones = ivy.ones(shape, dtype=x.dtype, device=device, out=out)
    return x * ones
