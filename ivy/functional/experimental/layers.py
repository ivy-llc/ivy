from typing import Optional, Union, Tuple, Literal
import ivy
from ivy.func_wrapper import (
    handle_array_like,
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
    >>> x = ivy.arange(12).reshape((2, 1, 3, 2))
    >>> print(ivy.avg_pool2d(x, (2, 2), (1, 1), 'SAME'))
    ivy.array([[[[ 1.,  2.],
             [ 3.,  4.],
             [ 4.,  5.]]],


           [[[ 7.,  8.],
             [ 9., 10.],
             [10., 11.]]]])
    >>> x = ivy.arange(48).reshape((2, 4, 3, 2))
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
@handle_array_like
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


@handle_exceptions
@to_native_arrays_and_back
@handle_array_like
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
    """
    return ivy.current_backend(x).dropout1d(
        x, prob, training=training, data_format=data_format, out=out
    )
