from typing import (
    Optional,
    Union,
    Tuple,
    Iterable,
    Callable,
    Literal,
    Any,
)
from numbers import Number
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    outputs_to_ivy_arrays,
    integer_arrays_to_float,
)
from ivy.exceptions import handle_exceptions
from math import sqrt, pi, cos


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def vorbis_window(
    window_length: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an array that contains a vorbis power complementary window
    of size window_length.

    Parameters
    ----------
    window_length
        the length of the vorbis window.
    dtype
        data type of the returned array. By default float32.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Input array with the vorbis window.

    Examples
    --------
    >>> ivy.vorbis_window(3)
    ivy.array([0.38268346, 1. , 0.38268352])

    >>> ivy.vorbis_window(5)
    ivy.array([0.14943586, 0.8563191 , 1. , 0.8563191, 0.14943568])
    """
    return ivy.current_backend().vorbis_window(window_length, dtype=dtype, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Generate a Hann window. The Hanning window
    is a taper formed by using a weighted cosine.

    Parameters
    ----------
    window_length
        the size of the returned window.
    periodic
        If True, returns a window to be used as periodic function.
        If False, return a symmetric window.
    dtype
        The data type to produce. Must be a floating point type.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array containing the window.

    Functional Examples
    -------------------
    >>> ivy.hann_window(4, True)
    ivy.array([0. , 0.5, 1. , 0.5])

    >>> ivy.hann_window(7, False)
    ivy.array([0.  , 0.25, 0.75, 1.  , 0.75, 0.25, 0.  ])

    """
    return ivy.current_backend().hann_window(
        window_length, periodic, dtype=dtype, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def max_pool2d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 2-D max pool given 4-D input x.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]*.
    kernel
        Size of the kernel i.e., the sliding window for each
        dimension of input. *[h,w]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list
        indicating the per-dimensio paddings.
    data_format
        NHWC" or "NCHW". Defaults to "NHWC".
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The result of the pooling operation.

    Both the description and the type hints above assumes an array input
    for simplicity, but this function is *nestable*, and therefore
    also accepts :class:`ivy.Container` instances in place of any of
    the arguments.

    Examples
    --------
    >>> x = ivy.arange(12).reshape((2, 1, 3, 2))
    >>> print(ivy.max_pool2d(x, (2, 2), (1, 1), 'SAME'))
    ivy.array([[[[ 2,  3],
     [ 4,  5],
     [ 4,  5]]],
    [[[ 8,  9],
     [10, 11],
     [10, 11]]]])

    >>> x = ivy.arange(48).reshape((2, 4, 3, 2))
    >>> print(ivy.max_pool2d(x, 3, 1, 'VALID'))
    ivy.array([[[[16, 17]],
    [[22, 23]]],
    [[[40, 41]],
    [[46, 47]]]])
    """
    return ivy.current_backend(x).max_pool2d(x, kernel, strides, padding, out=out)


@handle_out_argument
@to_native_arrays_and_back
@handle_out_argument
def max_pool1d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D max pool given 3-D input x.

    Parameters
    ----------
    x
        Input image *[batch_size, w, d_in]*.
    kernel
        Size of the kernel i.e., the sliding window for each
        dimension of input. *[w]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list
        indicating the per-dimension paddings.
    data_format
        NWC" or "NCW". Defaults to "NWC".
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The result of the pooling operation.

    Both the description and the type hints above assumes an array input
    for simplicity, but this function is *nestable*, and therefore
    also accepts :class:`ivy.Container` instances in place of any of
    the arguments.

    Examples
    --------
    >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
    >>> print(ivy.max_pool1d(x, 2, 2, 'SAME'))
    ivy.array([[[ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]],

           [[16., 17., 18., 19.],
            [20., 21., 22., 23.]]])
    >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
    >>> print(ivy.max_pool1d(x, 2, 2, 'VALID'))
    ivy.array([[[ 4.,  5.,  6.,  7.]],

       [[16., 17., 18., 19.]]])
    """
    return ivy.current_backend(x).max_pool1d(
        x, kernel, strides, padding, data_format=data_format, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Kaiser window with window length window_length and shape beta

    Parameters
    ----------
    window_length
        an int defining the length of the window.
    periodic
        If True, returns a periodic window suitable for use in spectral analysis.
        If False, returns a symmetric window suitable for use in filter design.
    beta
        a float used as shape parameter for the window.
    dtype
        data type of the returned array.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array containing the window.

    Examples
    --------
    >>> ivy.kaiser_window(5)
    ivy.array([5.2773e-05, 1.0172e-01, 7.9294e-01, 7.9294e-01, 1.0172e-01]])
    >>> ivy.kaiser_window(5, True, 5)
    ivy.array([0.0367, 0.4149, 0.9138, 0.9138, 0.4149])
    >>> ivy.kaiser_window(5, False, 5)
    ivy.array([0.0367, 0.5529, 1.0000, 0.5529, 0.0367])
    """
    return ivy.current_backend().kaiser_window(
        window_length, periodic, beta, dtype=dtype, out=out
    )


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


@outputs_to_ivy_arrays
@handle_out_argument
@handle_nestable
@handle_exceptions
def kaiser_bessel_derived_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Kaiser bessel derived window with
    window length window_length and shape beta

    Parameters
    ----------
    window_length
        an int defining the length of the window.
    periodic
        If True, returns a periodic window suitable for use in spectral analysis.
        If False, returns a symmetric window suitable for use in filter design.
    beta
        a float used as shape parameter for the window.
    dtype
        data type of the returned array
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array containing the window.

    Functional Examples
    -------------------
    >>> ivy.kaiser_bessel_derived_window(5)
    ivy.array([0.00713103, 0.70710677, 0.99997455, 0.99997455, 0.70710677])

    >>> ivy.kaiser_derived_window(5, False)
    ivy.array([0.00726415, 0.9999736 , 0.9999736 , 0.00726415])

    >>> ivy.kaiser_derived_window(5, False, 5)
    ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208])
    """
    window_length = window_length // 2
    w = ivy.kaiser_window(window_length + 1, periodic, beta)

    sum_i_N = sum([w[i] for i in range(0, window_length + 1)])

    def sum_i_n(n):
        return sum([w[i] for i in range(0, n + 1)])

    dn_low = [sqrt(sum_i_n(i) / sum_i_N) for i in range(0, window_length)]

    def sum_2N_1_n(n):
        return sum([w[i] for i in range(0, 2 * window_length - n)])

    dn_mid = [
        sqrt(sum_2N_1_n(i) / sum_i_N) for i in range(window_length, 2 * window_length)
    ]

    return ivy.array(dn_low + dn_mid, dtype=dtype, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def hamming_window(
    window_length: int,
    /,
    *,
    periodic: Optional[bool] = True,
    alpha: Optional[float] = 0.54,
    beta: Optional[float] = 0.46,
    dtype: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Hamming window with window length window_length

    Parameters
    ----------
    window_length
        an int defining the length of the window.
    periodic
         If True, returns a window to be used as periodic function.
         If False, return a symmetric window.
    alpha
        The coefficient alpha in the hamming window equation
    beta
        The coefficient beta in the hamming window equation
    dtype
        data type of the returned array.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array containing the window.

    Examples
    --------
    >>> ivy.hamming_window(5)
    ivy.array([0.0800, 0.3979, 0.9121, 0.9121, 0.3979])
    >>> ivy.hamming_window(5, periodic=False)
    ivy.array([0.0800, 0.5400, 1.0000, 0.5400, 0.0800])
    >>> ivy.hamming_window(5, periodic=False, alpha=0.2, beta=2)
    ivy.array([-1.8000,  0.2000,  2.2000,  0.2000, -1.8000])
    """
    if window_length == 0:
        return ivy.array([])
    elif window_length == 1:
        return ivy.array([1])
    else:
        if periodic is True:
            window_length = window_length + 1
            return ivy.array(
                [
                    alpha - beta * cos((2 * n * pi) / (window_length - 1))
                    for n in range(0, window_length)
                ][:-1],
                dtype=dtype,
                out=out,
            )
        else:
            return ivy.array(
                [
                    alpha - beta * cos((2 * n * pi) / (window_length - 1))
                    for n in range(0, window_length)
                ],
                dtype=dtype,
                out=out,
            )


@to_native_arrays_and_back
@integer_arrays_to_float
@handle_out_argument
@handle_nestable
@handle_exceptions
def dct(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Computes the 1D Discrete Cosine Tranformation of a given signal.

    Parameters
    ----------
    x
        The input signal.
    type
        The type of the dct. Must be 1, 2, 3 or 4.
    n
        The lenght of the transform. If n is less than the input signal lenght,
        then x is truncated, if n is larger then x is zero-padded.
    axis
        The axis to compute the DCT along.
    norm
        The type of normalization to be applied. Must be either None or "ortho".
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array containing the transformed input.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([8, 16, 24, 32, 40, 48, 56, 64])
    >>> ivy.dct(x, type=2, n=None, norm='ortho')
    ivy.array([102., -51.5, 0., -5.39, 0., -1.61, 0., -0.406])

    >>> x = ivy.array([[[8, 16, 24, 32], [40, 48, 56, 64]], 
               [[1,  2,  3,  4], [ 5,  6,  7,  8]]])
    >>> ivy.dct(x, type=1, n=None, axis=0, norm=None)
    ivy.array([[[ 9., 18., 27., 36.],
                [45., 54., 63., 72.]],
               [[ 7., 14., 21., 28.],
                [35., 42., 49., 56.]]])

    >>> x = ivy.array([[ 8.1, 16.2, 24.3, 32.4],
    ...                [40.5, 48.6, 56.7, 64.8]])
    >>> y = ivy.zeros((2, 4), dtype=ivy.float32)
    >>> ivy.dct(x, type=1, n=None, norm=None, out=y)
    >>> print(y)
    ivy.array([[ 1.22e+02, -3.24e+01,  1.91e-06, -8.10e+00],
               [ 3.16e+02, -3.24e+01,  3.81e-06, -8.10e+00]])

    >>> x = ivy.array([8., 16., 24., 32., 40., 48., 56., 64.])
    >>> ivy.dct(x, type=4, n=None, norm=None, out=x)
    >>> print(x)
    ivy.array([ 279. , -280. ,  128. , -115. ,   83.7,  -79.5,   69.8,  -68.7])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
    ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
    >>> ivy.dct(x, type=3, n=None, norm='ortho')
    {
        a: ivy.array([79.5, -70.4, 30., -23.6, 13.9, -10.1, 5.2, -1.95]),
        b: ivy.array([9.94, -8.8, 3.75, -2.95, 1.74, -1.26, 0.65, -0.244])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
    ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
    >>> container_n = ivy.Container(a=9, b=4)
    >>> container_type = ivy.Container(a=2, b=1)
    >>> container_norm = ivy.Container(a="ortho", b=None)
    >>> ivy.dct(x, type=container_type, n=container_n, norm=container_norm)
    {
        a: ivy.array([96., -28.2, -31.9, 22.9, -26., 19.8, -17., 10.9,
                    -5.89]),
        b: ivy.array([15., -4., 0., -1.])
    }
    """
    return ivy.current_backend().dct(x, type=type, n=n, axis=axis, norm=norm, out=out)
