from typing import (
    Optional,
    Union,
    Tuple,
    Iterable,
    Sequence,
    Generator,
    Callable,
    Any,
    Literal
)
from numbers import Number
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def flatten(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
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

    Returns
    -------
    ret
        the flattened array over the specified dimensions.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.concat.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = np.array([1,2], [3,4])
    >>> ivy.flatten(x)
    ivy.array([1, 2, 3, 4])

    >>> x = np.array(
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
    if start_dim == end_dim and len(x.shape) != 0:
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
    return ivy.reshape(x, tuple(lst))


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
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
def vstack(arrays: Sequence[ivy.Array], /) -> ivy.Array:
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
    return ivy.current_backend(arrays[0]).vstack(arrays)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def hstack(arrays: Sequence[ivy.Array], /) -> ivy.Array:
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
    return ivy.current_backend(arrays[0]).hstack(arrays)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
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


def _scatter_at_0_axis(input, value, start=None, end=None):
    dim_length = input.shape[0]
    if start is None:
        start = 0
    elif start < 0:
        start += dim_length
    if end is None:
        end = dim_length
    elif end < 0:
        end += dim_length
    i = 0
    value = ivy.asarray(value, dtype=input.dtype)
    if len(value.shape) > 1:
        value = ivy.flatten(value)
    for ind in ivy.ndindex(input.shape):
        if (ind[0] < end) and (ind[0] >= start):
            if len(value.shape) >= 1:
                input[ind] = value[i]
            else:
                input[ind] = value
            i += 1
    return input


def _set_pad_area(padded, width_pair, value_pair):
    padded = _scatter_at_0_axis(padded, value_pair[0], end=width_pair[0])
    padded = _scatter_at_0_axis(
        padded, value_pair[1], start=padded.shape[0] - width_pair[1]
    )
    return padded


def _get_edges(padded, width_pair):
    left_index = width_pair[0]
    left_edge = padded[left_index : left_index + 1, ...]
    right_index = padded.shape[0] - width_pair[1]
    right_edge = padded[right_index - 1 : right_index, ...]
    return left_edge, right_edge


def _get_linear_ramps(padded, width_pair, end_value_pair):
    edge_pair = _get_edges(padded, width_pair)
    left_ramp, right_ramp = (
        ivy.linspace(
            end_value,
            edge.squeeze(0),
            num=width,
            endpoint=False,
            dtype=padded.dtype,
            axis=0,
        )
        for end_value, edge, width in zip(end_value_pair, edge_pair, width_pair)
    )
    right_ramp = ivy.flip(right_ramp)
    return left_ramp, right_ramp


def _get_stats(padded, width_pair, length_pair, stat_func):
    left_index = width_pair[0]
    right_index = padded.shape[0] - width_pair[1]
    max_length = right_index - left_index
    left_length, right_length = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length
    left_chunk = padded[left_index : left_index + left_length, ...]
    left_stat = stat_func(left_chunk, axis=0, keepdims=True)
    if left_length == right_length == max_length:
        return left_stat, left_stat
    right_chunk = padded[right_index - right_length : right_index, ...]
    right_stat = stat_func(right_chunk, axis=0, keepdims=True)
    return left_stat, right_stat


def _set_reflect_both(padded, width_pair, method, include_edge=False):
    left_pad, right_pad = width_pair
    old_length = padded.shape[0] - right_pad - left_pad
    if include_edge:
        edge_offset = 1
    else:
        edge_offset = 0
        old_length -= 1
    if left_pad > 0:
        chunk_length = min(old_length, left_pad)
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_chunk = ivy.flip(padded[stop:start, ...])
        if method == "odd":
            left_chunk = 2 * padded[left_pad : left_pad + 1, ...] - left_chunk
        start = left_pad - chunk_length
        stop = left_pad
        padded = _scatter_at_0_axis(padded, left_chunk, start=start, end=stop)
        left_pad -= chunk_length
    if right_pad > 0:
        chunk_length = min(old_length, right_pad)
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_chunk = ivy.flip(padded[stop:start, ...])
        if method == "odd":
            right_chunk = 2 * padded[-right_pad - 1 : -right_pad, ...] - right_chunk
        start = padded.shape[0] - right_pad
        stop = start + chunk_length
        padded = _scatter_at_0_axis(padded, right_chunk, start=start, end=stop)
        right_pad -= chunk_length
    return left_pad, right_pad, padded


def _set_wrap_both(padded, width_pair):
    left_pad, right_pad = width_pair
    period = padded.shape[0] - right_pad - left_pad
    new_left_pad = 0
    new_right_pad = 0
    if left_pad > 0:
        right_chunk = padded[
            -right_pad - min(period, left_pad) : -right_pad if right_pad != 0 else None,
            ...,
        ]
        if left_pad > period:
            padded = _scatter_at_0_axis(
                padded, right_chunk, start=left_pad - period, end=left_pad
            )
            new_left_pad = left_pad - period
        else:
            padded = _scatter_at_0_axis(padded, right_chunk, end=left_pad)
    if right_pad > 0:
        left_chunk = padded[left_pad : left_pad + min(period, right_pad), ...]
        if right_pad > period:
            padded = _scatter_at_0_axis(
                padded, left_chunk, start=-right_pad, end=-right_pad + period
            )
            new_right_pad = right_pad - period
        else:
            padded = _scatter_at_0_axis(padded, left_chunk, start=-right_pad)
    return new_left_pad, new_right_pad, padded


def _to_pairs(x, n):
    if ivy.isscalar(x):
        return ((x, x),) * n
    elif ivy.asarray(x).shape == (2,):
        return ((x[0], x[1]),) * n
    ivy.assertions.check_equal(
        ivy.asarray(x).shape,
        (n, 2),
        message="values should be an integer or an iterable "
        "of ndim pairs where ndim is the number of "
        "the input's dimensions",
    )
    return x


def _pad_simple(array, pad_width, fill_value=None):
    new_shape = tuple(
        left + size + right for size, (left, right) in zip(array.shape, pad_width)
    )
    padded = ivy.zeros(new_shape, dtype=array.dtype)
    if fill_value is not None:
        padded = ivy.ones_like(padded) * fill_value
    sl = []
    for size, (left, right) in zip(array.shape, pad_width):
        sl.append(ivy.arange(left, left + size))
    if len(array.shape) > 1:
        array = ivy.flatten(array)
    j = 0
    for ind in ivy.ndindex(padded.shape):
        flag = True
        for i, k in enumerate(ind):
            if ivy.argwhere(sl[i] - k).shape[0] == sl[i].shape[0]:
                flag = False
                break
        if flag:
            padded[ind] = array[j]
            j += 1
    return padded


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
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
    constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
    end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
    reflect_type: Optional[Literal["even", "odd"]] = "even",
    out: Optional[ivy.Array] = None,
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
               function should modify a rank 1 array in-place following a signature
               like `padding_func(vector, iaxis_pad_width, iaxis, kwargs)`, where:
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
    out
        optional output array, for writing the result to. It must have a shape that
        the input broadcasts to.

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
    >>> y = ivy.pad(x, padding, mode="constant")
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
    input = ivy.asarray(input, dtype=input.dtype)
    pad_width = _to_pairs(pad_width, input.ndim)
    if callable(mode):
        func = mode
        padded = _pad_simple(input, pad_width, fill_value=0)
        for axis in range(padded.ndim):
            view = ivy.moveaxis(padded, axis, -1)
            inds = ivy.ndindex(view.shape[:-1])
            for ind in inds:
                view[ind] = func(view[ind], pad_width[axis], axis, kwargs)
        return padded
    padded = _pad_simple(input, pad_width)
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
            padded = _set_pad_area(padded, width_pair, value_pair)
            padded = ivy.moveaxis(padded, 0, -1)
    elif mode == "empty":
        pass
    elif mode == "edge":
        for axis, width_pair in zip(axes, pad_width):
            edge_pair = _get_edges(padded, width_pair)
            padded = _set_pad_area(padded, width_pair, edge_pair)
            padded = ivy.moveaxis(padded, 0, -1)
    elif mode == "linear_ramp":
        end_values = _to_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, end_values):
            ramp_pair = _get_linear_ramps(padded, width_pair, value_pair)
            padded = _set_pad_area(padded, width_pair, ramp_pair)
            padded = ivy.moveaxis(padded, 0, -1)
    elif mode in stat_functions:
        func = stat_functions[mode]
        stat_length = _to_pairs(stat_length, padded.ndim)
        for axis, width_pair, length_pair in zip(axes, pad_width, stat_length):
            stat_pair = _get_stats(padded, width_pair, length_pair, func)
            padded = _set_pad_area(padded, width_pair, stat_pair)
            padded = ivy.moveaxis(padded, 0, -1)
    elif mode in {"reflect", "symmetric"}:
        include_edge = True if mode == "symmetric" else False
        for axis, (left_index, right_index) in zip(axes, pad_width):
            if input.shape[0] == 1 and (left_index > 0 or right_index > 0):
                edge_pair = _get_edges(padded, (left_index, right_index))
                padded = _set_pad_area(padded, (left_index, right_index), edge_pair)
                continue
            while left_index > 0 or right_index > 0:
                left_index, right_index, padded = _set_reflect_both(
                    padded, (left_index, right_index), reflect_type, include_edge
                )
            padded = ivy.moveaxis(padded, 0, -1)
    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width):
            while left_index > 0 or right_index > 0:
                left_index, right_index, padded = _set_wrap_both(
                    padded, (left_index, right_index)
                )
            padded = ivy.moveaxis(padded, 0, -1)
    return padded
