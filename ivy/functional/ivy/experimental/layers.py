# global
import math
import itertools
from typing import Optional, Union, Tuple, Literal, Sequence

# local
import ivy
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
)
from ivy.utils.exceptions import handle_exceptions


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
    padding: Union[str, int, Tuple[int], Tuple[int, int]],
    /,
    *,
    data_format: str = "NHWC",
    dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
    ceil_mode: bool = False,
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
        indicating the per-dimension paddings.
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
    return ivy.current_backend(x).max_pool2d(
        x,
        kernel,
        strides,
        padding,
        data_format=data_format,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=out,
    )


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


@handle_nestable
@handle_exceptions
@to_native_arrays_and_back
@handle_array_like_without_promotion
def dropout3d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Randomly zero out entire channels with probability prob using samples from
     a Bernoulli distribution and the remaining channels are scaled by (1/1-prob).
     In this case, dropout3d performs a channel-wise dropout but assumes
     a channel is a 1D feature map.

    Parameters
    ----------
    x
        a 4D or 5D input array. Should have a floating-point data type.
    prob
        probability of a channel to be zero-ed.
    training
        controls whether dropout3d is performed during training or ignored
        during testing.
    data_format
        "NDHWC" or "NCDHW". Defaults to "NDHWC".
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

    """
    return ivy.current_backend(x).dropout3d(
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
    out: Optional[ivy.Array] = None,
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
    ivy.utils.assertions.check_equal(
        len(weights.shape), 2, message="weights must be 2-d"
    )

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


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
@handle_nestable
def dft(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = 1,
    inverse: bool = False,
    onesided: bool = False,
    dft_length: Optional[Union[int, Tuple[int]]] = None,
    norm: Optional[str] = "backward",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
        Computes the discrete Fourier transform of input.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*,
        where d_in indicates the dimension that needs FFT.
    axis
        The axis on which to perform the DFT. By default this
        value is  set to 1, which corresponds to the first dimension
        after the batch index.
    inverse
        Whether to perform the inverse discrete fourier transform.
        By default this value is set to False.
    onesided
        If onesided is True, only values for w in [0, 1, 2, …, floor(n_fft/2) + 1]
        are returned because the real-to-complex Fourier transform satisfies the
        conjugate symmetry, i.e., X[m, w] = X[m,w]=X[m,n_fft-w]*. Note if the
        input or window tensors are complex, then onesided output is not possible.
        Enabling onesided with real inputs performs a Real-valued fast Fourier
        transform (RFFT). When invoked with real or complex valued input, the
        default value is False. Values can be True or False.
    dft_length
        The length of the signal.If greater than the axis dimension,
        the signal will be zero-padded up to dft_length. If less than
        the axis dimension, only the first dft_length values will be
        used as the signal. It’s an optional value.
    norm
          Optional argument, "backward", "ortho" or "forward". Defaults to be
          "backward".
          "backward" indicates no normalization.
          "ortho" indicates normalization by 1/sqrt(n).
          "forward" indicates normalization by 1/n.
    out
        Optional output array, for writing the result to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The Fourier Transform of the input vector.If onesided is False,
        the following shape is expected: [batch_idx][signal_dim1][signal_dim2]
        …[signal_dimN][2]. If axis=0 and onesided is True, the following shape
        is expected: [batch_idx][floor(signal_dim1/2)+1][signal_dim2]…[signal_dimN][2].
        If axis=1 and onesided is True, the following shape is expected:
        [batch_idx][signal_dim1][floor(signal_dim2/2)+1]…[signal_dimN][2].
        If axis=N-1 and onesided is True, the following shape is expected:
        [batch_idx][signal_dim1][signal_dim2]…[floor(signal_dimN/2)+1][2].
        The signal_dim at the specified axis is equal to the dft_length.

    """
    if inverse:
        res = ifft(x, axis, norm=norm, n=dft_length, out=out)
    else:
        res = fft(x, axis, norm=norm, n=dft_length, out=out)

    if onesided:
        slices = [slice(0, a) for a in res.shape]
        slices[axis] = slice(0, res.shape[axis] // 2 + 1)
        res = res[tuple(slices)]
    return res


@to_native_arrays_and_back
@handle_exceptions
@handle_out_argument
@handle_nestable
def interp(x, xp, fp, left=None, right=None, period=None):
    x_arr = ivy.array(x)
    fix_later = False
    if x_arr.shape == ():
        x_arr = ivy.array([x])
        fix_later = True
    x = ivy.astype(x_arr, "float64")
    xp = ivy.astype(ivy.array(xp), "float64")
    fp = ivy.astype(ivy.array(fp), "float64")
    ivy.utils.assertions.check_equal(xp.ndim, 1)
    ivy.utils.assertions.check_equal(fp.ndim, 1)
    ivy.utils.assertions.check_equal(xp.shape[0], fp.shape[0])
    if period is not None:
        ivy.utils.assertions.check_equal(period, 0, inverse=True)
        period = ivy.abs(period)
        x = ivy.remainder(x, period)
        xp = ivy.remainder(xp, period)
        asort_xp = ivy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = ivy.concat((xp[-1:] - period, xp, xp[0:1] + period))
        fp = ivy.concat((fp[-1:], fp, fp[0:1]))

    def interp_inner(value):
        value = ivy.array(value)
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


def _fill_triangle_kernel(x):
    return ivy.maximum(0, 1 - ivy.abs(x))


def compute_weight_mat(
    input_size, output_size, scale, align_corners, kernel_fn, antialias: bool
):
    inv_scale = 1.0 / scale
    kernel_scale = ivy.maximum(inv_scale, 1.0) if antialias else 1.0
    if not align_corners or align_corners is None:
        sample_f = (ivy.arange(output_size) + 0.5) * inv_scale - 0.5
        x = ivy.abs(sample_f[None, :] - ivy.arange(input_size)[:, None]) / kernel_scale
    else:
        sample_f = ivy.linspace(0, input_size - 1, output_size)
        x = ivy.abs(sample_f[None, :] - ivy.arange(input_size)[:, None]) / (
            kernel_scale
        )
    weights = kernel_fn(x)
    total_weight_sum = ivy.sum(weights, axis=0, keepdims=True)
    weights = ivy.where(
        ivy.abs(total_weight_sum) > 1000.0 * float(ivy.finfo("float32").eps),
        ivy.divide(weights, ivy.where(total_weight_sum != 0, total_weight_sum, 1)),
        0,
    )
    input_size_minus_0_5 = input_size if align_corners else input_size - 0.5
    return ivy.where(
        ivy.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)[None, :],
        weights,
        0,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def interpolate(
    x: Union[ivy.Array, ivy.NativeArray],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Union[
        Literal["linear", "bilinear", "trilinear", "nearest", "area", "nearest_exact"]
    ] = "linear",
    align_corners: Optional[bool] = None,
    antialias: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Down/up samples the input to the given size.
    The algorithm used for interpolation is determined by mode.

    Parameters
    ----------
    x
        Input array, Must have the shape
        [batch x channels x [optional depth] x [optional height] x width].
    size
        Output size.
    mode
        Interpolation mode. Can be one of the following:
        - linear
        - bilinear
        - trilinear
        - nearest
        - area
    align_corners
        If True, the corner pixels of the input and output tensors are aligned,
        and thus preserving the values at the corner pixels. If False, the corner
        pixels are not aligned, and the interpolation uses edge value padding for
        out-of-boundary values.
        only has an effect when mode is 'linear', 'bilinear',
        'bicubic' or 'trilinear'. Default: False
    antialias
        If True, antialiasing is applied when downsampling an image.
        Supported modes: 'bilinear', 'bicubic'.
    out
        Optional output array, for writing the result to. It must
        have a shape that the inputs broadcast to.

    Returns
    -------
        resized array

    """
    dims = len(x.shape) - 2
    size = (size,) * dims if isinstance(size, int) else tuple(size)
    spatial_dims = [2 + i for i in range(dims)]
    input_shape = ivy.shape(x)
    scale = [ivy.divide(size[i], input_shape[spatial_dims[i]]) for i in range(dims)]
    if mode == "bilinear" or mode == "linear" or mode == "trilinear":
        if mode == "linear":
            equation = "ijk,km->ijm"
        elif mode == "bilinear":
            equation = "ijkl,km,ln->ijmn"
        elif mode == "trilinear":
            equation = "ijklm,kn,lo,mp->ijnop"
        output_shape = tuple(input_shape[:2]) + size
        operands = []
        for i, d in enumerate(spatial_dims):
            m = input_shape[d]
            n = output_shape[d]
            w = compute_weight_mat(
                m, n, scale[i], align_corners, _fill_triangle_kernel, antialias
            ).astype(x.dtype)
            operands.append(w)
        ret = ivy.einsum(equation, x, *operands)
    elif mode == "nearest" or mode == "nearest_exact":
        ret = ivy.zeros((x.shape[:2] + tuple(size)))
        for i, ba in enumerate(x):
            for j, ch in enumerate(ba):
                w_scale = size[-1] / x.shape[-1]
                if dims == 3:
                    h_scale = size[-2] / x.shape[-2]
                    d_scale = size[-3] / x.shape[-3]
                    for d_dim in range(size[0]):
                        for h_dim in range(size[1]):
                            for w_dim in range(size[2]):
                                ret[i, j, d_dim, h_dim, w_dim] = x[i][j][
                                    round(d_dim // d_scale)
                                ][round(h_dim // h_scale)][round(w_dim // w_scale)]
                elif dims == 2:
                    h_scale = size[-2] / x.shape[-2]
                    for h_dim in range(size[0]):
                        for w_dim in range(size[1]):
                            ret[i, j, h_dim, w_dim] = x[i][j][round(h_dim // h_scale)][
                                round(w_dim // w_scale)
                            ]
                elif dims == 1:
                    for w_dim in range(size[0]):
                        ret[i, j, w_dim] = x[i][j][round(w_dim // w_scale)]
    elif mode == "area":
        ret = ivy.zeros((x.shape[:2] + size))
        scale = ivy.divide(ivy.shape(x)[2:], size)
        for i, ba in enumerate(x):
            for j, ch in enumerate(ba):
                if dims == 3:
                    for d_dim in range(size[0]):
                        for h_dim in range(size[1]):
                            for w_dim in range(size[2]):
                                d_index = (
                                    int(d_dim * scale[0]),
                                    math.ceil((d_dim + 1) * scale[0]),
                                )
                                h_index = (
                                    int(h_dim * scale[1]),
                                    math.ceil((h_dim + 1) * scale[1]),
                                )
                                w_index = (
                                    int(w_dim * scale[2]),
                                    math.ceil((w_dim + 1) * scale[2]),
                                )
                                scale_z = d_index[1] - d_index[0]
                                scale_y = h_index[1] - h_index[0]
                                scale_x = w_index[1] - w_index[0]
                                area = scale_z * scale_y * scale_x
                                ret[i, j, d_dim, h_dim, w_dim] = ivy.sum(
                                    ch[
                                        d_index[0] : d_index[1],
                                        h_index[0] : h_index[1],
                                        w_index[0] : w_index[1],
                                    ]
                                ) * (1 / area)
                elif dims == 2:
                    for h_dim in range(size[0]):
                        for w_dim in range(size[1]):
                            h_index = (
                                int(h_dim * scale[0]),
                                math.ceil((h_dim + 1) * scale[0]),
                            )
                            w_index = (
                                int(w_dim * scale[1]),
                                math.ceil((w_dim + 1) * scale[1]),
                            )
                            scale_y = h_index[1] - h_index[0]
                            scale_x = w_index[1] - w_index[0]
                            area = scale_y * scale_x
                            ret[i, j, h_dim, w_dim] = ivy.sum(
                                ch[h_index[0] : h_index[1], w_index[0] : w_index[1]]
                            ) * (1 / area)
                else:
                    for w_dim in range(size[0]):
                        w_index = (
                            int(w_dim * scale[0]),
                            math.ceil((w_dim + 1) * scale[0]),
                        )
                        scale_x = w_index[1] - w_index[0]
                        ret[i, j, w_dim] = ivy.sum(ch[w_index[0] : w_index[1]]) * (
                            1 / scale_x
                        )
    return ivy.astype(ret, ivy.dtype(x), out=out)


interpolate.mixed_function = True


def _output_ceil_shape(w, f, p, s):
    return math.ceil((w - f + p) / s) + 1


def _padding_ceil_mode(w, f, p, s):
    remaining_pixels = (w - f + sum(p)) % s
    if s > 1 and remaining_pixels != 0 and f > 1:
        input_size = w + sum(p)
        # making sure that the remaining pixels are supposed
        # to be covered by the window
        # they won't be covered if stride is big enough to skip them
        if input_size - remaining_pixels - (f - 1) + s > input_size:
            return p
        output_shape = _output_ceil_shape(
            w,
            f,
            sum(p),
            s,
        )
        # calculating new padding with ceil_output_shape
        new_pad = (output_shape - 1) * s + f - w
        # updating pad_list with new padding by adding it to the end
        p = (
            p[0],
            p[1] + new_pad - sum(p),
        )
    return p


def _compute_idx(in_size, out_size, device):
    out_range = ivy.arange(out_size, device=device, dtype=ivy.int64)
    i0 = ivy.trunc_divide(out_range * in_size, out_size).astype(ivy.int64)
    maxlength = in_size // out_size + 1
    in_size_mod = in_size % out_size
    # adaptive = True iff there are kernels with different lengths
    adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
    if adaptive:
        maxlength += 1
    elif in_size_mod == 0:
        maxlength -= 1
    range_max = ivy.arange(maxlength, device=device, dtype=ivy.int64)
    idx = ivy.expand_dims(i0, axis=-1) + range_max
    if adaptive:
        maxval = ivy.full_like(idx, fill_value=in_size - 1)
        idx = ivy.minimum(idx, maxval)
        i1 = ivy.trunc_divide(
            (out_range + 1) * in_size + out_size - 1, out_size
        ).astype(ivy.int64)
        length = i1 - i0
    else:
        length = maxlength
    return idx, length, range_max, adaptive


def _expand_to_dim(x, dim):
    for _ in range(dim - len(x.shape)):
        x = ivy.expand_dims(x, axis=-1)
    return x


def _mask(vals, length, range_max, dim):
    if isinstance(length, int):
        return vals, length
    else:
        assert dim < 0
        mask = ivy.greater_equal(range_max, ivy.expand_dims(length, axis=-1))
        if dim == -2:
            mask = _expand_to_dim(mask, 4)
        vals = ivy.where(mask, 0.0, vals)
        length = _expand_to_dim(length, -dim)
        return vals, length


def adaptive_avg_pool1d(
        input: Union[ivy.Array, ivy.NativeArray],
        output_size: int,
) -> ivy.Array:
    """
    Applies a 1D adaptive average pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
    input
        Input array. Must have shape (N, C, L_in) or (C, L_in) where N is
        the batch dimension, C is the feature dimension, and L_in is the spatial
        dimension.
    output_size
        Spatial output size.

    Returns
    -------
        The result of the pooling operation. Will have shape (N, C, L_out) or
        (C, L_out), where L_out = `output_size`

    """
    squeeze = False
    if len(input.shape) == 2:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 3:
        raise ivy.utils.exceptions.IvyException(
            f"Got {len(input.shape)}D input, but only 2D and 3D inputs are supported.",
        )

    if input.shape[-1] % output_size == 0:
        stride = input.shape[-1] // output_size
        kernel_size = input.shape[-1] - (output_size - 1) * stride
        pooled_output = ivy.avg_pool1d(
            input, kernel_size, stride, "VALID", data_format="NCW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output

    idxw, length_w, range_max_w, adaptive_w = _compute_idx(
        input.shape[-1], output_size, input.device
    )

    # to numpy and back in order to bypass a slicing error in tensorflow
    vals = ivy.array(input.to_numpy()[..., idxw])

    if not adaptive_w:
        return ivy.mean(vals, axis=-1)

    vals, length_w = _mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i in range(vals.shape[-1]):
        if ret is None:
            ret = vals[..., i]
        else:
            ret = ret + vals[..., i]
    pooled_output = ret / length_w

    if squeeze:
        return ivy.squeeze(pooled_output, axis=0)
    return pooled_output


def adaptive_avg_pool2d(
        input: Union[ivy.Array, ivy.NativeArray],
        output_size: Union[Sequence[int], int],
) -> ivy.Array:
    """
    Applies a 2D adaptive average pooling over an input signal composed of several input
    planes.

    Parameters
    ----------
    input
        Input array. Must have shape (N, C, H_in, W_in) or (C, H_in, W_in) where N is
        the batch dimension, C is the feature dimension, and H_in and W_in are the 2
        spatial dimensions.
    output_size
        Spatial output size.

    Returns
    -------
        The result of the pooling operation. Will have shape (N, C, S_0, S_1) or
        (C, S_0, S_1), where S = `output_size`

    """
    squeeze = False
    if len(input.shape) == 3:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif len(input.shape) != 4:
        raise ivy.utils.exceptions.IvyException(
            f"Got {len(input.shape)}D input, but only 3D and 4D inputs are supported.",
        )

    if all(i_s % o_s == 0 for i_s, o_s in zip(input.shape[-2:], output_size)):
        stride = tuple(i_s // o_s for i_s, o_s in zip(input.shape[-2:], output_size))
        kernel_size = tuple(
            i_s - (o_s - 1) * st
            for i_s, o_s, st in zip(input.shape[-2:], output_size, stride)
        )
        pooled_output = ivy.avg_pool2d(
            input, kernel_size, stride, "VALID", data_format="NCHW"
        )
        if squeeze:
            return ivy.squeeze(pooled_output, axis=0)
        return pooled_output

    idxh, length_h, range_max_h, adaptive_h = _compute_idx(
        input.shape[-2], output_size[-2], input.device
    )
    idxw, length_w, range_max_w, adaptive_w = _compute_idx(
        input.shape[-1], output_size[-1], input.device
    )

    # to numpy and back in order to bypass a slicing error in tensorflow
    vals = ivy.array(input.to_numpy()[..., _expand_to_dim(idxh, 4), idxw])

    if not adaptive_h and not adaptive_w:
        return ivy.mean(vals, axis=(-3, -1))

    vals, length_h = _mask(vals, length_h, range_max_h, dim=-2)
    vals, length_w = _mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i, j in itertools.product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    pooled_output = ret / (length_h * length_w)

    if squeeze:
        return ivy.squeeze(pooled_output, axis=0)
    return pooled_output
