# global
from __future__ import annotations
from math import sqrt, pi, cos
from typing import Union, Tuple, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    infer_device,
    outputs_to_ivy_arrays,
    handle_nestable,
    to_native_arrays_and_back,
    handle_out_argument,
)


@outputs_to_ivy_arrays
@infer_device
@handle_nestable
@handle_exceptions
def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Tuple[ivy.Array]:
    """Returns the indices of the upper triangular part of a row by col matrix in a
    2-by-N shape (tuple of two N dimensional arrays), where the first row contains
    row coordinates of all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.  The upper triangular part
    of the matrix is defined as the elements on and above the diagonal.  The argument
    k controls which diagonal to consider. If k = 0, all elements on and above the main
    diagonal are retained. A positive value excludes just as many diagonals above the
    main diagonal, and similarly a negative value includes just as many diagonals
    below the main diagonal. The main diagonal are the set of indices
    {(i,i)} for i∈[0,min{n_rows, n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
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
       k > 0 moves upwards and k < 0 moves downwards
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
    >>> x = ivy.triu_indices(4,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    ivy.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,1)
    >>> print(x)
    (ivy.array([0, 0, 0, 1, 1, 2]),
    ivy.array([1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,-2)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
    ivy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,2,0)
    >>> print(x)
    (ivy.array([0, 0, 1]),
    ivy.array([0, 1, 1]))

    >>> x = ivy.triu_indices(2,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1]),
    ivy.array([0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,-4,0)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(4,4,100)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(2,4,-100)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1]), ivy.array([0, 1, 2, 3, 0, 1, 2, 3]))

    """
    return current_backend().triu_indices(n_rows, n_cols, k, device=device)


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
