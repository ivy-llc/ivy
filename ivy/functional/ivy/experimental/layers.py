from typing import Optional, Union, Tuple, Literal, Sequence
import ivy
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
)
from ivy.exceptions import handle_exceptions


@handle_nestable
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
    >>> x = ivy.arange(12.).reshape((2, 1, 3, 2))
    >>> print(ivy.max_pool2d(x, (2, 2), (1, 1), 'SAME'))
    ivy.array([[[[ 2,  3],
             [ 4,  5],
             [ 4,  5]]],


           [[[ 8,  9],
             [10, 11],
             [10, 11]]]])

    >>> x = ivy.arange(48.).reshape((2, 4, 3, 2))
    >>> print(ivy.max_pool2d(x, 3, 1, 'VALID'))
    ivy.array([[[[16, 17]],

            [[22, 23]]],


           [[[40, 41]],

            [[46, 47]]]])
    """
    return ivy.current_backend(x).max_pool2d(x, kernel, strides, padding, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def max_pool3d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 3-D max pool given 5-D input x.

    Parameters
    ----------
    x
        Input volume *[batch_size,d,h,w,d_in]*.
    kernel
        Convolution filters *[d,h,w]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NDHWC" or "NCDHW". Defaults to "NDHWC".
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

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
    >>> x = ivy.arange(48.).reshape((2, 3, 2, 2, 2))
    >>> print(ivy.max_pool3d(x, 2, 2, 'VALID'))
    ivy.array([[[[[14., 15.]]]],



       [[[[38., 39.]]]]])
    >>> print(ivy.max_pool3d(x, 2, 2, 'SAME'))
    ivy.array([[[[[14., 15.]]],


        [[[22., 23.]]]],



       [[[[38., 39.]]],


        [[[46., 47.]]]]])

    """
    return ivy.current_backend(x).max_pool3d(
        x, kernel, strides, padding, data_format=data_format, out=out
    )


@handle_nestable
@to_native_arrays_and_back
@handle_out_argument
def avg_pool1d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D avg pool given 3-D input x.

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
    >>> print(ivy.avg_pool1d(x, 2, 2, 'SAME'))
    ivy.array([[[ 2.,  3.,  4.,  5.],
            [ 8.,  9., 10., 11.]],

           [[14., 15., 16., 17.],
            [20., 21., 22., 23.]]])
    >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
    >>> print(ivy.avg_pool1d(x, 2, 2, 'VALID'))
    ivy.array([[[ 2.,  3.,  4.,  5.]],

           [[14., 15., 16., 17.]]])
    """
    return ivy.current_backend(x).avg_pool1d(
        x, kernel, strides, padding, data_format=data_format, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def avg_pool2d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 2-D average pool given 4-D input x.

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
    >>> x = ivy.arange(12.).reshape((2, 1, 3, 2))
    >>> print(ivy.avg_pool2d(x, (2, 2), (1, 1), 'SAME'))
    ivy.array([[[[ 1.,  2.],
             [ 3.,  4.],
             [ 4.,  5.]]],


           [[[ 7.,  8.],
             [ 9., 10.],
             [10., 11.]]]])
    >>> x = ivy.arange(48.).reshape((2, 4, 3, 2))
    >>> print(ivy.avg_pool2d(x, 3, 1, 'VALID'))
    ivy.array([[[[ 8.,  9.]],

        [[14., 15.]]],


       [[[32., 33.]],

        [[38., 39.]]]])

    """
    return ivy.current_backend(x).avg_pool2d(
        x, kernel, strides, padding, data_format=data_format, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def avg_pool3d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 3-D avg pool given 5-D input x.

    Parameters
    ----------
    x
        Input volume *[batch_size,d,h,w,d_in]*.
    kernel
        Convolution filters *[d,h,w]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NDHWC" or "NCDHW". Defaults to "NDHWC".
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

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
    >>> x = ivy.arange(48.).reshape((2, 3, 2, 2, 2))
    >>> print(ivy.avg_pool3d(x,2,2,'VALID'))
    ivy.array([[[[[ 7.,  8.]]]],



           [[[[31., 32.]]]]])
    >>> print(ivy.avg_pool3d(x,2,2,'SAME'))
    ivy.array([[[[[ 7.,  8.]]],


            [[[19., 20.]]]],



           [[[[31., 32.]]],


            [[[43., 44.]]]]])

    """
    return ivy.current_backend(x).avg_pool3d(
        x, kernel, strides, padding, data_format=data_format, out=out
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
    ...                [[1,  2,  3,  4], [ 5,  6,  7,  8]]])
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
    return ivy.current_backend(x).dct(x, type=type, n=n, axis=axis, norm=norm, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
@handle_array_like_without_promotion
def fft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    /,
    *,
    norm: Optional[str] = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Computes the one dimensional discrete Fourier transform given input at least
    1-D input x.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*,
        where d_in indicates the dimension that needs FFT.
    dim
        The dimension along which to take the one dimensional FFT.
    norm
        Optional argument, "backward", "ortho" or "forward". Defaults to be "backward".
        "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$.
        "forward" indicates normalization by $\frac{1}{n}$.
    n
        Optional argument indicating the sequence length, if given, the input would be
        padded with zero or truncated to length n before performing FFT.
        Should be a integer greater than 1.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the FFT operation.

    Examples
    --------
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0)
    ivy.array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
            2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
            9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
            1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0, n=16)
    ivy.array([-3.44509285e-16+1.14423775e-17j,  1.00000000e+00+5.02733949e+00j,
        8.00000000e+00-8.11483250e-16j,  1.00000000e+00-5.02733949e+00j,
        2.33486982e-16+1.22464680e-16j,  1.00000000e+00-1.49660576e+00j,
        0.00000000e+00+1.22464680e-16j,  1.00000000e+00-6.68178638e-01j,
        9.95799250e-17+2.33486982e-16j,  1.00000000e+00-1.98912367e-01j,
        0.00000000e+00+7.66951701e-17j,  1.00000000e+00+1.98912367e-01j,
        1.14423775e-17+1.22464680e-16j,  1.00000000e+00+6.68178638e-01j,
        0.00000000e+00+1.22464680e-16j,  1.00000000e+00+1.49660576e+00j])
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0, norm="ortho")
    ivy.array([-1.21802426e-16+4.04549134e-18j,  2.82842712e+00-2.86902654e-16j,
        8.25501143e-17+4.32978028e-17j,  0.00000000e+00+4.32978028e-17j,
        3.52068201e-17+8.25501143e-17j,  0.00000000e+00+2.71158374e-17j,
        4.04549134e-18+4.32978028e-17j,  0.00000000e+00+4.32978028e-17j])
    """
    return ivy.current_backend(x).fft(x, dim, norm=norm, n=n, out=out)


@handle_nestable
@handle_exceptions
@to_native_arrays_and_back
@handle_array_like_without_promotion
def dropout1d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Randomly zero out entire channels with probability prob using samples from
     a Bernoulli distribution and the remaining channels are scaled by (1/1-prob).
     In this case, dropout1d performs a channel-wise dropout but assumes
     a channel is a 1D feature map.

    Parameters
    ----------
    x
        a 2D or 3D input array. Should have a floating-point data type.
    prob
        probability of a channel to be zero-ed.
    training
        controls whether dropout1d is performed during training or ignored
        during testing.
    data_format
        "NWC" or "NCW". Defaults to "NWC".
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        an array with some channels zero-ed and the rest of channels are
         scaled by (1/1-prob).

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 1, 1]).reshape([1, 1, 3])
    >>> y = ivy.dropout1d(x, 0.5)
    >>> print(y)
    ivy.array([[[2., 0, 2.]]])

    >>> x = ivy.array([1, 1, 1]).reshape([1, 1, 3])
    >>> y = ivy.dropout1d(x, 1, training=False, data_format="NCW")
    >>> print(y)
    ivy.array([[[1, 1, 1]]])

    With one :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([100, 200, 300]).reshape([1, 1, 3]),
    ...                   b=ivy.array([400, 500, 600]).reshape([1, 1, 3]))
    >>> y = ivy.dropout1d(x, 0.5)
    >>> print(y)
    {
    a: ivy.array([[[200., 400., 0.]]]),
    b: ivy.array([[[0., 0., 0.]]])
    }
    """
    return ivy.current_backend(x).dropout1d(
        x, prob, training=training, data_format=data_format, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
def ifft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    *,
    norm: Optional[str] = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Computes the one dimensional discrete Fourier transform given input at least
    1-D input x.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*,
        where d_in indicates the dimension that needs IFFT.
    dim
        The dimension along which to take the one dimensional IFFT.
    norm
        Optional argument, "backward", "ortho" or "forward". Defaults to be "backward".
        "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$.
        "forward" indicates normalization by $\frac{1}{n}$.
    n
        Optional argument indicating the sequence length, if given, the input would be
        padded with zero or truncated to length n before performing IFFT.
        Should be a integer greater than 1.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the IFFT operation.

    Examples
    --------
    >>> ivy.ifft(np.exp(2j * np.pi * np.arange(8) / 8), 0)
    ivy.array([-4.30636606e-17+1.43029718e-18j,  0.00000000e+00+1.53080850e-17j,
                1.43029718e-18+1.53080850e-17j,  0.00000000e+00+9.58689626e-18j,
                1.24474906e-17+2.91858728e-17j,  0.00000000e+00+1.53080850e-17j,
                2.91858728e-17+1.53080850e-17j,  1.00000000e+00-1.01435406e-16j])
    >>> ivy.ifft(np.exp(2j * np.pi * np.arange(8) / 8), 0, n=16)
    ivy.array([-2.15318303e-17+7.15148591e-19j,  6.25000000e-02+9.35378602e-02j,
                0.00000000e+00+7.65404249e-18j,  6.25000000e-02+4.17611649e-02j,
                7.15148591e-19+7.65404249e-18j,  6.25000000e-02+1.24320230e-02j,
                0.00000000e+00+4.79344813e-18j,  6.25000000e-02-1.24320230e-02j,
                6.22374531e-18+1.45929364e-17j,  6.25000000e-02-4.17611649e-02j,
                0.00000000e+00+7.65404249e-18j,  6.25000000e-02-9.35378602e-02j,
                1.45929364e-17+7.65404249e-18j,  6.25000000e-02-3.14208718e-01j,
                5.00000000e-01-5.07177031e-17j,  6.25000000e-02+3.14208718e-01j])
    >>> ivy.ifft(np.exp(2j * np.pi * np.arange(8) / 8), 0, norm="ortho")
    ivy.array([-1.21802426e-16+4.04549134e-18j,  0.00000000e+00+4.32978028e-17j,
                4.04549134e-18+4.32978028e-17j,  0.00000000e+00+2.71158374e-17j,
                3.52068201e-17+8.25501143e-17j,  0.00000000e+00+4.32978028e-17j,
                8.25501143e-17+4.32978028e-17j,  2.82842712e+00-2.86902654e-16j])
    """
    return ivy.current_backend(x).ifft(x, dim, norm=norm, n=n, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
@handle_nestable
def embedding(
    weights: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    max_norm: Optional[int] = None,
    out=None,
) -> ivy.Array:
    """Embeds a given tensor of indices using a given tensor of weights.

    Parameters
    ----------
    weights
        The weights tensor.
    indices
        The indices tensor.
    max_norm
        The maximum norm of the embeddings.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the embedding operation.

    Examples
    --------
    >>> weights = ivy.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> indices = ivy.array([0, 2])
    >>> print(ivy.embedding(weights, indices, max_norm=5))
    ivy.array([[1., 2., 3.],
                [7., 8., 9.]])
    """
    ivy.assertions.check_equal(len(weights.shape), 2, message="weights must be 2-d")

    ret = ivy.empty(
        indices.shape + (weights.shape[1],), dtype=ivy.as_ivy_dtype(weights.dtype)
    )
    if not ivy.is_ivy_array(indices):
        indices = ivy.array(indices, dtype=ivy.int32)

    for i, x in ivy.ndenumerate(indices):

        if ivy.exists(max_norm):
            ret[i] = ivy.clip_vector_norm(weights[x, :], max_norm)
        else:
            ret[i] = weights[x, :]
    return ret


def interp(x, xp, fp, left=None, right=None, period=None):
    x_arr = ivy.array(x)
    fix_later = False
    if x_arr.shape == ():
        x_arr = ivy.array([x])
        fix_later = True
    x = ivy.astype(x_arr, "float64")
    xp = ivy.astype(ivy.array(xp), "float64")
    fp = ivy.astype(ivy.array(fp), "float64")
    ivy.assertions.check_equal(xp.ndim, 1)
    ivy.assertions.check_equal(fp.ndim, 1)
    ivy.assertions.check_equal(xp.shape[0], fp.shape[0])
    if period is not None:
        ivy.assertions.check_equal(period, 0, inverse=True)
        period = ivy.abs(period)
        x = ivy.remainder(x, period)
        xp = ivy.remainder(xp, period)
        asort_xp = ivy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = ivy.concat((xp[-1:] - period, xp, xp[0:1] + period))
        fp = ivy.concat((fp[-1:], fp, fp[0:1]))

    def interp_inner(value):
        if value < xp[0]:
            return left if left is not None else fp[0]
        elif value > xp[-1]:
            return right if right is not None else fp[-1]
        else:
            last = None
            if xp.shape[0] < 3:
                for i in range(xp.shape[0] - 1, -1, -1):
                    if xp[i] == value:
                        return fp[i]
                    elif xp[i] < value:
                        last = i
            else:
                first = 0
                last = xp.shape[0]
                while first < last:
                    midpoint = (first + last) // 2
                    if xp[midpoint] == value:
                        already_exists = ivy.argwhere(xp == value)
                        if already_exists.shape[0] > 0:
                            return fp[already_exists[-1][0]]
                        return fp[midpoint]
                    else:
                        if value < xp[midpoint]:
                            last = midpoint - 1
                        else:
                            first = midpoint + 1
            dist = (value - xp[last]) / (xp[last + 1] - xp[last])
            return (fp[last + 1] - fp[last]) * dist + fp[last]

    ret = ivy.map(interp_inner, unique={"value": x})
    if fix_later:
        return ivy.astype(ivy.array(ret[0]), "float64")
    else:
        return ivy.astype(ivy.array(ret), "float64")


def interpolate(
    x: Union[ivy.Array, ivy.NativeArray],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Union[Literal["linear", "bilinear", "trilinear"]] = "linear",
    align_corners: Optional[bool] = True,
    antialias: Optional[bool] = False,
):
    """

    Parameters
    ----------
    x
        Input array, Must have the shape
        [batch x channels x [optional depth] x [optional height] x width].
    size
    mode
    align_corners
    antialias

    Returns
    -------

    """
    if mode == "linear":
        size = size[0] if isinstance(size, (list, tuple)) else size
        if not align_corners:
            x_up = ivy.arange(0, ivy.shape(x)[-1])
            missing = (ivy.arange(0, size) + 0.5) * (ivy.shape(x)[-1] / size) - 0.5
        else:
            x_up = ivy.linspace(0, 1, ivy.shape(x)[-1])
            missing = ivy.linspace(0, 1, size)
        ret = ivy.zeros(ivy.shape(x)[:-1] + (size,))
        for i, ba in enumerate(x):
            for j, ch in enumerate(ba):
                ret[i][j] = ivy.interp(missing, x_up, ch)
    elif mode == "bilinear":
        if not align_corners:
            x_up_h = ivy.arange(0, ivy.shape(x)[-2])
            x_up_w = ivy.arange(0, ivy.shape(x)[-1])
            missing_h = (ivy.arange(0, size[0]) + 0.5) * (
                ivy.shape(x)[-2] / size[0]
            ) - 0.5
            missing_w = (ivy.arange(0, size[1]) + 0.5) * (
                ivy.shape(x)[-1] / size[1]
            ) - 0.5
        else:
            x_up_h = ivy.linspace(0, 1, ivy.shape(x)[-2])
            x_up_w = ivy.linspace(0, 1, ivy.shape(x)[-1])
            missing_h = ivy.linspace(0, 1, size[0])
            missing_w = ivy.linspace(0, 1, size[1])
        ret = ivy.zeros(ivy.shape(x)[:-2] + (size[1], size[0]))
        for i, ba in enumerate(x):
            for j, ch in enumerate(ba):
                row_ret = ivy.zeros((ivy.shape(x)[-2], size[1]))
                for k, row in enumerate(ch):
                    row_ret[k] = ivy.interp(missing_w, x_up_w, row)
                row_ret = row_ret.T
                for k, col in enumerate(row_ret):
                    ret[i][j][k] = ivy.interp(missing_h, x_up_h, col)
        ret = ivy.permute_dims(ret, (0, 1, 3, 2))
    elif mode == "trilinear":
        if not align_corners:
            x_up_d = ivy.arange(0, ivy.shape(x)[-3])
            x_up_h = ivy.arange(0, ivy.shape(x)[-2])
            x_up_w = ivy.arange(0, ivy.shape(x)[-1])
            missing_d = (ivy.arange(0, size[0]) + 0.5) * (
                ivy.shape(x)[-3] / size[0]
            ) - 0.5
            missing_h = (ivy.arange(0, size[1]) + 0.5) * (
                ivy.shape(x)[-2] / size[1]
            ) - 0.5
            missing_w = (ivy.arange(0, size[2]) + 0.5) * (
                ivy.shape(x)[-1] / size[2]
            ) - 0.5
        else:
            x_up_d = ivy.linspace(0, 1, ivy.shape(x)[-3])
            x_up_h = ivy.linspace(0, 1, ivy.shape(x)[-2])
            x_up_w = ivy.linspace(0, 1, ivy.shape(x)[-1])
            missing_d = ivy.linspace(0, 1, size[0])
            missing_h = ivy.linspace(0, 1, size[1])
            missing_w = ivy.linspace(0, 1, size[2])
        ret = ivy.zeros(ivy.shape(x)[:-3] + (size[1], size[2], size[0]))
        for i, ba in enumerate(x):
            for j, ch in enumerate(ba):
                depth_ret = ivy.zeros((x.shape[-3], size[2], size[1]))
                row_ret = ivy.zeros((ivy.shape(x)[-3], ivy.shape(x)[-2], size[2]))
                for k, depth in enumerate(ch):
                    for (
                        l,
                        row,
                    ) in enumerate(ch[k]):
                        row_ret[k][l] = ivy.interp(missing_w, x_up_w, row)
                row_ret = row_ret.transpose((0, 2, 1))
                for k, row in enumerate(ch):
                    for (
                        l,
                        col,
                    ) in enumerate(row_ret[k]):
                        depth_ret[k][l] = ivy.interp(missing_h, x_up_h, col)
                depth_ret = depth_ret.transpose((2, 1, 0))
                for k, col in enumerate(depth_ret):
                    for (
                        l,
                        depth,
                    ) in enumerate(depth_ret[k]):
                        ret[i][j][k][l] = ivy.interp(missing_d, x_up_d, depth)
        ret = ret.transpose((0, 1, 4, 2, 3))
    return ivy.astype(ret, ivy.dtype(x))
