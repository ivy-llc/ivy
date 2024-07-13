# global
from typing import Union, Tuple, Optional, Sequence, Iterable, Generator
import warnings

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    outputs_to_ivy_arrays,
    handle_nestable,
    to_native_arrays_and_back,
    handle_out_argument,
    infer_dtype,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_device,
    handle_backend_invalid,
    handle_array_function,
)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device
def vorbis_window(
    window_length: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return an array that contains a vorbis power complementary window of
    size window_length.

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


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device
def hann_window(
    size: int,
    *,
    periodic: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Generate a Hann window. The Hanning window is a taper formed by using a
    weighted cosine.

    Parameters
    ----------
    size
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

    Examples
    --------
    >>> ivy.hann_window(4, periodic = True)
    ivy.array([0. , 0.5, 1. , 0.5])

    >>> ivy.hann_window(7, periodic = False)
    ivy.array([0.  , 0.25, 0.75, 1.  , 0.75, 0.25, 0.  ])
    """
    return ivy.current_backend().hann_window(
        size, periodic=periodic, dtype=dtype, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device
def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Kaiser window with window length window_length and shape
    beta.

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


@handle_exceptions
@handle_nestable
@handle_out_argument
@infer_dtype
def kaiser_bessel_derived_window(
    window_length: int,
    beta: float = 12.0,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Kaiser bessel derived window with window length
    window_length and shape beta.

    Parameters
    ----------
    window_length
        an int defining the length of the window.
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

    Examples
    --------
    >>> ivy.kaiser_bessel_derived_window(5)
    ivy.array([0.00726415, 0.9999736 , 0.9999736 , 0.00726415])

    >>> ivy.kaiser_bessel_derived_window(5, 5)
    ivy.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208])
    """
    if window_length < 2:
        result = ivy.array([], dtype=dtype)
        if ivy.exists(out):
            ivy.inplace_update(out, result)
        return result
    half_len = window_length // 2
    kaiser_w = ivy.kaiser_window(half_len + 1, False, beta, dtype=dtype)
    kaiser_w_csum = ivy.cumsum(kaiser_w)
    half_w = ivy.sqrt(kaiser_w_csum[:-1] / kaiser_w_csum[-1:])
    window = ivy.concat((half_w, half_w[::-1]), axis=0)
    result = window.astype(dtype)
    return result


@handle_exceptions
@handle_nestable
@infer_dtype
def hamming_window(
    window_length: int,
    *,
    periodic: bool = True,
    alpha: float = 0.54,
    beta: float = 0.46,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Hamming window with window length window_length.

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
    if window_length < 2:
        return ivy.ones([window_length], dtype=dtype, out=out)
    if periodic:
        count = ivy.arange(window_length) / window_length
    else:
        count = ivy.linspace(0, window_length, window_length)
    result = (alpha - beta * ivy.cos(2 * ivy.pi * count)).astype(dtype)
    if ivy.exists(out):
        result = ivy.inplace_update(out, result)
    return result


hamming_window.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "handle_out_argument",
        "handle_device",
    ),
    "to_skip": (),
}


@handle_exceptions
@handle_nestable
@outputs_to_ivy_arrays
@handle_device
def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Tuple[ivy.Array, ...]:
    """Return the indices of the lower triangular part of a row by col matrix
    in a 2-by-N shape (tuple of two N dimensional arrays), where the first row
    contains row coordinates of all indices and the second row contains column
    coordinates. Indices are ordered based on rows and then columns.  The lower
    triangular part of the matrix is defined as the elements on and below the
    diagonal.  The argument k controls which diagonal to consider. If k = 0,
    all elements on and below the main diagonal are retained. A positive value
    excludes just as many diagonals below the main diagonal, and similarly a
    negative value includes just as many diagonals above the main diagonal. The
    main diagonal are the set of indices {(i,i)} for i∈[0,min{n_rows,
    n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html
    for examples

    Tensorflow does not support slicing 2-D tensor with tuple of tensor of indices

    Parameters
    ----------
    n_rows
       number of rows in the 2-d matrix.
    n_cols
       number of columns in the 2-d matrix. If None n_cols will be the same as n_rows
    k
       number of shifts from the main diagonal. k = 0 includes main diagonal,
       k > 0 moves downward and k < 0 moves upward
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an 2xN shape, tuple of two N dimensional, where first subarray (i.e. ret[0])
        contains row coordinates of all indices and the second subarray (i.e ret[1])
        contains columns indices.

    Function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.tril_indices(4,4,0)
    >>> print(x)
    (ivy.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
    ivy.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    >>> x = ivy.tril_indices(4,4,1)
    >>> print(x)
    (ivy.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    ivy.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]))

    >>> x = ivy.tril_indices(4,4,-2)
    >>> print(x)
    (ivy.array([2, 3, 3]), ivy.array([0, 0, 1]))

    >>> x = ivy.tril_indices(4,2,0)
    >>> print(x)
    (ivy.array([0, 1, 1, 2, 2, 3, 3]),
    ivy.array([0, 0, 1, 0, 1, 0, 1]))

    >>> x = ivy.tril_indices(2,4,0)
    >>> print(x)
    (ivy.array([0, 1, 1]), ivy.array([0, 0, 1]))

    >>> x = ivy.tril_indices(4,-4,0)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.tril_indices(4,4,100)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    ivy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))

    >>> x = ivy.tril_indices(2,4,-100)
    >>> print(x)
    (ivy.array([]), ivy.array([]))
    """
    return current_backend().tril_indices(n_rows, n_cols, k, device=device)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_ivy_arrays
@infer_dtype
@handle_device
def eye_like(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    k: int = 0,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a 2D array filled with ones on the k diagonal and zeros
    elsewhere. having the same ``shape`` as the first and last dim of input
    array ``x``. input array ``x`` should to be 2D.

    Parameters
    ----------
    x
         input array from which to derive the output array shape.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and 0 to the main diagonal. Default: ``0``.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        the device on which to place the created array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``ones`` in
        diagonal ``k`` and ``zeros`` elsewhere.


    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances as a replacement to any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([[0, 1],[2, 3]])
    >>> y1 = ivy.eye_like(x1)
    >>> print(y1)
    ivy.array([[1., 0.],
               [0., 1.]])

    >>> x1 = ivy.array([[0, 1, 2],[3, 4, 5],[6, 7, 8]])
    >>> y1 = ivy.eye_like(x1, k=1)
    >>> print(y1)
    ivy.array([[0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 0.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[3, 8],[0, 2]]), b=ivy.array([[0, 2], [8, 5]]))
    >>> y = x.eye_like()
    >>> print(y)
    {
        a: ivy.array([[1., 0.],
                      [0., 1.]]),
        b: ivy.array([[1., 0.],
                      [0., 1.]])
    }
    """
    shape = ivy.shape(x, as_array=True)
    dim = len(shape)
    if dim <= 1:
        cols = dim
    else:
        cols = int(shape[-1])
    rows = 0 if dim < 1 else int(shape[0])
    return ivy.eye(
        rows,
        cols,
        k=k,
        dtype=dtype,
        device=device,
        out=out,
    )


def _iter_product(*args, repeat=1):
    # itertools.product
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


@handle_exceptions
@inputs_to_ivy_arrays
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

    def _ndenumerate(input):
        if ivy.is_ivy_array(input) and input.shape == ():
            yield (), ivy.to_scalar(input)
        else:
            i = [range(k) for k in input.shape]
            for idx in _iter_product(*i):
                yield idx, input[idx]

    input = input if ivy.is_ivy_array(input) else ivy.array(input)
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
    args = [range(k) for k in shape]
    return _iter_product(*args)


@handle_exceptions
def indices(
    dimensions: Sequence[int],
    *,
    dtype: Union[ivy.Dtype, ivy.NativeDtype] = ivy.int64,
    sparse: bool = False,
) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
    """Return an array representing the indices of a grid.

    Parameters
    ----------
    dimensions
        The shape of the grid.
    dtype
        The data type of the result.
    sparse
        Return a sparse representation of the grid instead of a dense representation.

    Returns
    -------
    ret
        If sparse is False, returns one grid indices array of shape
        (len(dimensions),) + tuple(dimensions).
        If sparse is True, returns a tuple of arrays each of shape
        (1, ..., 1, dimensions[i], 1, ..., 1) with dimensions[i] in the ith place.

    Examples
    --------
    >>> ivy.indices((3, 2))
    ivy.array([[[0 0]
                [1 1]
                [2 2]]
               [[0 1]
                [0 1]
                [0 1]]])
    >>> ivy.indices((3, 2), sparse=True)
    (ivy.array([[0], [1], [2]]), ivy.array([[0, 1]]))
    """
    if sparse:
        return tuple(
            ivy.arange(dim)
            .expand_dims(
                axis=[j for j in range(len(dimensions)) if i != j],
            )
            .astype(dtype)
            for i, dim in enumerate(dimensions)
        )
    else:
        grid = ivy.meshgrid(*[ivy.arange(dim) for dim in dimensions], indexing="ij")
        return ivy.stack(grid, axis=0).astype(dtype)


indices.mixed_backend_wrappers = {
    "to_add": ("handle_device",),
    "to_skip": (),
}


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@to_native_arrays_and_back
def unsorted_segment_min(
    data: Union[ivy.Array, ivy.NativeArray],
    segment_ids: Union[ivy.Array, ivy.NativeArray],
    num_segments: Union[int, ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Compute the minimum along segments of an array. Segments are defined by
    an integer array of segment IDs.

    Note
    ----
    If the given segment ID `i` is negative, then the corresponding
    value is dropped, and will not be included in the result.

    Parameters
    ----------
    data
        The array from which to gather values.

    segment_ids
        Must be in the same size with the first dimension of `data`. Has to be
        of integer data type. The index-th element of `segment_ids` array is
        the segment identifier for the index-th element of `data`.

    num_segments
        An integer or array representing the total number of distinct segment IDs.

    Returns
    -------
    ret
        The output array, representing the result of a segmented min operation.
        For each segment, it computes the min value in `data` where `segment_ids`
        equals to segment ID.
    """
    return ivy.current_backend().unsorted_segment_min(data, segment_ids, num_segments)


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
def unsorted_segment_sum(
    data: Union[ivy.Array, ivy.NativeArray],
    segment_ids: Union[ivy.Array, ivy.NativeArray],
    num_segments: Union[int, ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Compute the sum of elements along segments of an array. Segments are
    defined by an integer array of segment IDs.

    Parameters
    ----------
    data
        The array from which to gather values.

    segment_ids
        Must be in the same size with the first dimension of `data`. Has to be
        of integer data type. The index-th element of `segment_ids` array is
        the segment identifier for the index-th element of `data`.

    num_segments
        An integer or array representing the total number of distinct segment IDs.

    Returns
    -------
    ret
        The output array, representing the result of a segmented sum operation.
        For each segment, it computes the sum of values in `data` where `segment_ids`
        equals to segment ID.
    """
    return ivy.current_backend().unsorted_segment_sum(data, segment_ids, num_segments)


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@infer_dtype
@handle_device
def blackman_window(
    size: int,
    *,
    periodic: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Generate a Blackman window. The Blackman window is a taper formed by
    using the first three terms of a summation of cosines. It was designed to
    have close to the minimal leakage possible. It is close to optimal, only
    slightly worse than a Kaiser window.

    Parameters
    ----------
    window_length
        the window_length of the returned window.
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

    Examples
    --------
    >>> ivy.blackman_window(4, periodic = True)
    ivy.array([-1.38777878e-17,  3.40000000e-01,  1.00000000e+00,  3.40000000e-01])
    >>> ivy.blackman_window(7, periodic = False)
    ivy.array([-1.38777878e-17,  1.30000000e-01,  6.30000000e-01,  1.00000000e+00,
        6.30000000e-01,  1.30000000e-01, -1.38777878e-17])
    """
    return ivy.current_backend().blackman_window(
        size, periodic=periodic, dtype=dtype, out=out
    )


@handle_exceptions
@handle_nestable
@infer_dtype
def random_tucker(
    shape: Sequence[int],
    rank: Sequence[int],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    full: Optional[bool] = False,
    orthogonal: Optional[bool] = False,
    seed: Optional[int] = None,
    non_negative: Optional[bool] = False,
) -> Union[ivy.TuckerTensor, ivy.Array]:
    """Generate a random Tucker tensor.

    Parameters
    ----------
    shape
        shape of the tensor to generate
    rank
        rank of the Tucker decomposition
        if int, the same rank is used for each mode
        otherwise, dimension of each mode
    full
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    orthogonal
        if True, creates a tensor with orthogonal components
    seed
        seed for generating random numbers
    non_negative


    Returns
    -------
        ivy.TuckerTensor
    """
    rank = ivy.TuckerTensor.validate_tucker_rank(shape, rank)

    if orthogonal:
        for i, (s, r) in enumerate(zip(shape, rank)):
            if r > s:
                warnings.warn(
                    "Selected orthogonal=True, but selected a rank larger than the"
                    f" tensor size for mode {{0}}: rank[{i}]={r} > shape[{i}]={s}."
                )

    factors = []
    for s, r in zip(shape, rank):
        if orthogonal:
            factor = ivy.random_uniform(shape=(s, s), seed=seed, dtype=dtype)
            Q, _ = ivy.qr(factor)
            factors.append(ivy.array(Q[:, :r]))
        else:
            factors.append(ivy.random_uniform(shape=(s, r), seed=seed, dtype=dtype))

    core = ivy.random_uniform(shape=rank, seed=seed, dtype=dtype)

    if non_negative:
        factors = [ivy.abs(f) for f in factors]
        core = ivy.abs(core)

    if full:
        return ivy.TuckerTensor.tucker_to_tensor((core, factors))
    else:
        return ivy.TuckerTensor((core, factors))


@handle_exceptions
@handle_nestable
@infer_dtype
def random_cp(
    shape: Sequence[int],
    rank: int,
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    full: Optional[bool] = False,
    orthogonal: Optional[bool] = False,
    seed: Optional[int] = None,
    normalise_factors: Optional[bool] = True,
) -> Union[ivy.CPTensor, ivy.Array]:
    """Generate a random CP tensor.

    Parameters
    ----------
    shape
        shape of the tensor to generate
    rank
        rank of the CP decomposition
    full
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    orthogonal
        if True, creates a tensor with orthogonal components
    seed
        seed for generating random numbers

    Returns
    -------
        ivy.CPTensor
    """
    rank = ivy.CPTensor.validate_cp_rank(shape, rank)
    if (rank > min(shape)) and orthogonal:
        warnings.warn(
            "Can only construct orthogonal tensors when rank <= min(shape) but got "
            f"a tensor with min(shape)={min(shape)} < rank={rank}"
        )

    factors = [
        (ivy.random_uniform(shape=(s, rank), dtype=dtype, seed=seed)) for s in shape
    ]
    weights = ivy.ones((rank,), dtype=dtype)
    if orthogonal:
        factors = [ivy.qr(factor)[0] for factor in factors]

    if full:
        return ivy.CPTensor.cp_to_tensor((weights, factors))
    elif normalise_factors:
        return ivy.CPTensor.cp_normalize((weights, factors))
    else:
        return ivy.CPTensor((weights, factors))


@handle_exceptions
@handle_nestable
@infer_dtype
def random_tr(
    shape: Sequence[int],
    rank: Sequence[int],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    full: Optional[bool] = False,
    seed: Optional[int] = None,
) -> Union[ivy.TRTensor, ivy.Array]:
    """Generate a random TR tensor.

    Parameters
    ----------
    shape : tuple
        shape of the tensor to generate
    rank : Sequence[int]
        rank of the TR decomposition
        must verify rank[0] == rank[-1] (boundary conditions)
        and len(rank) == len(shape)+1
    full : bool, optional, default is False
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    seed :
        seed for generating random numbers
    context : dict
        context in which to create the tensor

    Returns
    -------
    ivy.TRTensor or ivy.Array if full is True
    """
    rank = ivy.TRTensor.validate_tr_rank(shape, rank)
    # Make sure it's not a tuple but a list
    rank = list(rank)
    _check_first_and_last_rank_elements_are_equal(rank)
    factors = [
        ivy.random_uniform(shape=(rank[i], s, rank[i + 1]), dtype=dtype, seed=seed)
        for i, s in enumerate(shape)
    ]
    if full:
        return ivy.TRTensor.tr_to_tensor(factors)
    else:
        return ivy.TRTensor(factors)


def _check_first_and_last_rank_elements_are_equal(rank):
    if rank[0] != rank[-1]:
        message = (
            f"Provided rank[0] == {rank[0]} and rank[-1] == {rank[-1]} "
            "but boundary conditions dictate rank[0] == rank[-1]."
        )
        raise ValueError(message)


@handle_exceptions
@handle_nestable
@infer_dtype
def random_parafac2(
    shapes: Sequence[int],
    rank: int,
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    full: Optional[bool] = False,
    seed: Optional[int] = None,
    normalise_factors: Optional[bool] = True,
) -> Union[ivy.Parafac2Tensor, ivy.Array]:
    """Generate a random PARAFAC2 tensor.

    Parameters
    ----------
    shapes
        A shapes of the tensor to generate
    rank
        rank of the Parafac2 decomposition
    full
        if True, a full tensor is returned otherwise,
        the decomposed tensor is returned
     seed
        seed for generating random numbers

    Returns
    -------
      ivy.Parafac2Tensor
    """
    if any(shape[1] != shapes[0][1] for shape in shapes):
        raise ValueError("All matrices must have equal number of columns.")

    projection_matrices = [
        ivy.qr(ivy.random_uniform(shape=(shape[0], rank), dtype=dtype, seed=seed))[0]
        for shape in shapes
    ]
    weights, factors = ivy.random_cp(
        [len(shapes), rank, shapes[0][1]],
        rank,
        normalise_factors=False,
        seed=seed,
        dtype=dtype,
    )

    parafac2_tensor = ivy.Parafac2Tensor((weights, factors, projection_matrices))

    if normalise_factors:
        parafac2_tensor = ivy.Parafac2Tensor.parafac2_normalise(parafac2_tensor)

    if full:
        return ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor)
    else:
        return parafac2_tensor


@handle_exceptions
@handle_nestable
@infer_dtype
def random_tt(
    shape: Sequence[int],
    rank: Union[Sequence[int], int],
    /,
    *,
    full: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
) -> Union[ivy.TTTensor, ivy.Array]:
    """Generate a random TT/MPS tensor.

    Parameters
    ----------
    shape
        shape of the tensor to generate
    rank
        rank of the TT decomposition
        must verify rank[0] == rank[-1] ==1 (boundary conditions)
        and len(rank) == len(shape)+1
    full
        if True, a full tensor is returned
        otherwise, the decomposed tensor is returned
    seed
        seed for generating random numbers

    Returns
    -------
        ivy.TTTensor
    """
    rank = ivy.TTTensor.validate_tt_rank(shape, rank)

    rank = list(rank)
    if rank[0] != 1:
        message = (
            f"Provided rank[0] == {rank[0]} but boundaring conditions dictatate rank[0]"
            " == rank[-1] == 1."
        )
        raise ValueError(message)
    if rank[-1] != 1:
        message = (
            f"Provided rank[-1] == {rank[-1]} but boundaring conditions dictatate"
            " rank[0] == rank[-1] == 1."
        )
        raise ValueError(message)

    factors = [
        (ivy.random_uniform(shape=(rank[i], s, rank[i + 1]), dtype=dtype, seed=seed))
        for i, s in enumerate(shape)
    ]

    if full:
        return ivy.TTTensor.tt_to_tensor(factors)
    else:
        return ivy.TTTensor(factors)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def trilu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    upper: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return the upper or lower triangular part of a matrix
    (or a stack of matrices) ``x``.
     note::
        The upper triangular part of the matrix is defined as the elements
        on and above the specified diagonal ``k``. The lower triangular part
        of the matrix is defined as the elements on and below the specified
        diagonal ``k``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.    *,
    k
        diagonal below or above which to zero elements. If k = 0, the diagonal is the
        main diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    upper
        indicates whether upper or lower part of matrix is retained. Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the upper or lower triangular part(s). The returned array
        must have the same shape and data type as x. All elements below or above the
        specified diagonal k must be zeroed. The returned array should be allocated on
        the same device as x.
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).trilu(x, k=k, upper=upper, out=out)


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
def mel_weight_matrix(
    num_mel_bins: int,
    dft_length: int,
    sample_rate: int,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 3000.0,
):
    """Generate a MelWeightMatrix that can be used to re-weight a Tensor
    containing a linearly sampled frequency spectra (from DFT or STFT) into
    num_mel_bins frequency information based on the [lower_edge_hertz,
    upper_edge_hertz]

    range on the mel scale. This function defines the mel scale in terms of a frequency
    in hertz according to the following formula: mel(f) = 2595 * log10(1 + f/700)

    Parameters
    ----------
    num_mel_bins
        The number of bands in the mel spectrum.
    dft_length
        The size of the original DFT obtained from (n_fft / 2 + 1).
    sample_rate
        Samples per second of the input signal.
    lower_edge_hertz
        Lower bound on the frequencies to be included in the mel spectrum.
    upper_edge_hertz
        The desired top edge of the highest frequency band.

    Returns
    -------
    ret
        MelWeightMatrix of shape:  [frames, num_mel_bins].

    Examples
    --------
    >>> ivy.mel_weight_matrix(3,3,8000)
    ivy.array([[0.        ,0.        , 0.],
              [0.        ,0. , 0.75694758],
              [0.        ,0. , 0.       ]])
    """
    return ivy.current_backend().mel_weight_matrix(
        num_mel_bins,
        dft_length,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )


# unsorted_segment_mean
@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
def unsorted_segment_mean(
    data: Union[ivy.Array, ivy.NativeArray],
    segment_ids: Union[ivy.Array, ivy.NativeArray],
    num_segments: Union[int, ivy.Array, ivy.NativeArray],
) -> ivy.Array:
    """Compute the mean of elements along segments of an array. Segments are
    defined by an integer array of segment IDs.

    Parameters
    ----------
    data : Union[ivy.Array, ivy.NativeArray]
        The array from which to gather values.

    segment_ids : Union[ivy.Array, ivy.NativeArray]
        Must be in the same size with the first dimension of `data`. Has to be
        of integer data type. The index-th element of `segment_ids` array is
        the segment identifier for the index-th element of `data`.

    num_segments : Union[int, ivy.Array, ivy.NativeArray]
        An integer or array representing the total number of distinct segment IDs.

    Returns
    -------
    ivy.Array
        The output array, representing the result of a segmented mean operation.
        For each segment, it computes the mean value in `data` where `segment_ids`
        equals to segment ID.
    """
    return ivy.current_backend().unsorted_segment_mean(data, segment_ids, num_segments)


@handle_exceptions
@handle_nestable
@handle_array_function
@to_native_arrays_and_back
def polyval(
    coeffs: Union[ivy.Array, ivy.NativeArray],
    x: Union[ivy.Array, ivy.NativeArray],
):
    """Evaluate and return a polynomial at specific given values.

    Parameters
    ----------
    coeffs
        Polynomial coefficients (including zero) from highest degree to constant term.
    x
        The value of the indeterminate variable at which to evaluate the polynomial.

    Returns
    -------
    ret
       Simplified result of substituting x in the coefficients - final value
       of polynomial.

    Examples
    --------
    >>> ivy.polyval([3, 0, 1], 5)
    ivy.array(76)
    """
    return ivy.current_backend().polyval(
        coeffs,
        x,
    )
