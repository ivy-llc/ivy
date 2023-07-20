# global
import math
import itertools
from typing import Optional, Union, Tuple, Literal, Sequence, Callable
from functools import reduce as _reduce
import builtins

# local
import ivy
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_partial_mixed_function,
    inputs_to_ivy_arrays,
    handle_array_function,
    handle_device_shifting,
)
from ivy.functional.ivy.experimental.general import _correct_ivy_callable
from ivy.utils.exceptions import handle_exceptions

_min = builtins.min
_slice = builtins.slice
_max = builtins.max


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
    """
    Compute a 1-D max pool given 3-D input x.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def max_unpool1d(
    x: ivy.Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D max unpooling given the 1-D pooled input x and its indices.

    Parameters
    ----------
    x
        Pooled input image *[batch_size, w, d_in]*.
    indices
        Indices obtained from the corresponding max pooling operation.
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
        The result of the unpooling operation.

    Both the description and the type hints above assume an array input
    for simplicity, but this function is *nestable*, and therefore
    also accepts :class:`ivy.Container` instances in place of any of
    the arguments.

    Examples
    --------
    >>> x = ivy.arange(0, 24.).reshape((2, 3, 4))
    >>> pool_result = ivy.max_pool1d(x, 2, 2, 'SAME')
    >>> print(pool_result)
    ivy.array([[[ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]],

           [[16., 17., 18., 19.],
            [20., 21., 22., 23.]]])
    >>> unpool_result = ivy.max_unpool1d(pool_result, indices, 2, 2, 'SAME')
    >>> print(unpool_result)
    ivy.array([[[ 0.,  4.,  0.,  5.,  0.,  6.,  0.,  7.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  8.,  0.,  9.,  0., 10.,  0., 11.,  0.]],

           [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 16.,  0., 17.,  0.],
            [ 0., 18.,  0., 19.,  0.,  0.,  0.,  0., 20.,  0., 21.,  0.]]])
    """
    return ivy.current_backend(x).max_unpool1d(
        x, indices, kernel, strides, padding, data_format=data_format, out=out
    )


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
    """
    Compute a 2-D max pool given 4-D input x.

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


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
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
    """
    Compute a 3-D max pool given 5-D input x.

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
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def avg_pool1d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    division_override: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D avg pool given 3-D input x.

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
    count_include_pad
        Whether to include padding in the averaging calculation.
    ceil_mode
        Whether to use ceil or floor for creating the output shape.
    division_override
        If specified, it will be used as the divisor,
        otherwise kernel_size will be used.
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
        x,
        kernel,
        strides,
        padding,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        division_override=division_override,
        out=out,
    )


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def avg_pool2d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 2-D average pool given 4-D input x.

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
    count_include_pad
        Whether to include padding in the averaging calculation.
    ceil_mode
        Whether to use ceil or floor for creating the output shape.
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
        x,
        kernel,
        strides,
        padding,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        out=out,
    )


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def avg_pool3d(
    x: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    count_include_pad: bool = False,
    ceil_mode: bool = False,
    divisor_override: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 3-D avg pool given 5-D input x.

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
    count_include_pad
        Whether to include padding in the averaging calculation.
    ceil_mode
        Whether to use ceil or floor for creating the output shape.
    divisor_override
        If specified, it will be used as divisor, otherwise kernel_size will be used.
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
        x,
        kernel,
        strides,
        padding,
        data_format=data_format,
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        out=out,
    )


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def pool(
    x: Union[ivy.Array, ivy.NativeArray],
    window_shape: Union[int, Tuple[int], Tuple[int, int]],
    pool_type: str,
    /,
    *,
    strides: Optional[Union[int, Tuple[int], Tuple[int, int]]] = None,
    padding: str = "VALID",
    data_format: Optional[str] = None,
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = None,
    ceil_mode: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Perform an N-D pooling operation.

    Parameters
    ----------
    x
        Input array to pool over.
    window_shape
        Shape of the pooling window.
    pool_type
        Type of pooling operation, either 'MAX' or 'AVG'.
    strides
        Strides of the pooling operation.
    padding
        Padding type, either 'VALID' or 'SAME'.
    data_format
        Data format of the input and output data, either 'NCHW' or 'NHWC'.
    dilations
        Dilation rate of the pooling operation.
    ceil_mode
        Whether to use ceil or floor for creating the output shape.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the pooling operation.

    Examples
    --------
    >>> x = ivy.arange(12.).reshape((2, 1, 3, 2))
    >>> print(ivy.pool(x, (2, 2), 'MAX', (1, 1), 'SAME'))
    ivy.array([[[[ 1.,  2.],
                [ 3.,  4.],
                [ 4.,  5.]]],
            [[[ 7.,  8.],
                [ 9., 10.],
                [10., 11.]]]])
    >>> x = ivy.arange(48.).reshape((2, 4, 3, 2))
    >>> print(ivy.pool(x, 3, 'AVG', 1, 'VALID'))
    ivy.array([[[[ 8.,  9.]],
            [[14., 15.]]],
            [[[32., 33.]],
            [[38., 39.]]]])
    """
    return ivy.current_backend(x).pool(
        x,
        window_shape,
        pool_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        ceil_mode=ceil_mode,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def dct(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the 1D Discrete Cosine Tranformation of a given signal.

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


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def idct(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Compute the 1D Inverse Discrete Cosine Tranformation of a given signal.

    Parameters
    ----------
    x
        The input signal.
    type
        The type of the idct. Must be 1, 2, 3 or 4.
    n
        The length of the transform. If n is less than the input signal length,
        then x is truncated, if n is larger then x is zero-padded.
    axis
        The axis to compute the IDCT along.
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
    >>> ivy.idct(x, type=2, n=None, norm='ortho')
    ivy.array([ 79.49862671, -70.37691498,  30.00390816, -23.58938599,
        13.92713165, -10.078475  ,   5.19664812,  -1.95411837])

    >>> x = ivy.array([[[8, 16, 24, 32], [40, 48, 56, 64]],
    ...                [[1,  2,  3,  4], [ 5,  6,  7,  8]]])
    >>> ivy.idct(x, type=1, n=None, axis=0, norm=None)
    ivy.array([[[ 9., 18., 27., 36.],
        [45., 54., 63., 72.]],
       [[ 7., 14., 21., 28.],
        [35., 42., 49., 56.]]])

    >>> x = ivy.array([[ 8.1, 16.2, 24.3, 32.4],
    ...                [40.5, 48.6, 56.7, 64.8]])
    >>> y = ivy.zeros((2, 4), dtype=ivy.float32)
    >>> ivy.idct(x, type=1, n=None, norm=None, out=y)
    >>> print(y)
    ivy.array([[ 1.21500000e+02, -3.24000015e+01,  1.90734863e-06,
            -8.10000420e+00],
           [ 3.15899994e+02, -3.24000053e+01,  3.81469727e-06,
            -8.09999847e+00]])

    >>> x = ivy.array([8., 16., 24., 32., 40., 48., 56., 64.])
    >>> ivy.idct(x, type=4, n=None, norm=None, out=x)
    >>> print(x)
    ivy.array([ 279.4135742 , -279.6779785 ,  128.3770599 , -114.8719864 ,
             83.72109985,  -79.52869415,   69.79182434,  -68.72489166])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
    ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
    >>> ivy.idct(x, type=3, n=None, norm='ortho')
    {
        a: ivy.array([1.01823372e+02, -5.15385818e+01, 1.36371455e-06, -5.38763905e+00,
                      0., -1.60722279e+00, -8.80319249e-08, -4.05617893e-01]),
        b: ivy.array([1.27279215e+01, -6.44232273e+00, 1.70464318e-07, -6.73454881e-01,
                      0., -2.00902849e-01, -1.10039906e-08, -5.07022366e-02])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([8, 16, 24, 32, 40, 48, 56, 64]),
    ...                   b=ivy.array([1,  2,  3,  4,  5,  6,  7,  8]))
    >>> container_n = ivy.Container(a=9, b=4)
    >>> container_type = ivy.Container(a=2, b=1)
    >>> container_norm = ivy.Container(a="ortho", b=None)
    >>> ivy.idct(x, type=container_type, n=container_n, norm=container_norm)
    {
        a: ivy.array([86.29723358, -66.6950531, 9.93914509, 2.88008738,
                      -16.18951225, 18.06697273, -17.57439804, 11.68861485,
                      -4.41308832]),
        b: ivy.array([15., -4., -2.22044605e-16, -1.])
    }
    """
    return ivy.current_backend(x).idct(x, type=type, n=n, axis=axis, norm=norm, out=out)


idct.mixed_backend_wrappers = {"to_add": ("handle_device_shifting",), "to_skip": ()}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def fft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    /,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Compute the one dimensional discrete Fourier transform given input at least 1-D
    input x.

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
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def dropout1d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Randomly zero out entire channels with probability prob using samples from a
    Bernoulli distribution and the remaining channels are scaled by (1/1-prob). In this
    case, dropout1d performs a channel-wise dropout but assumes a channel is a 1D
    feature map.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def dropout2d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Randomly zero out entire channels with probability prob using samples from a
    Bernoulli distribution and the remaining channels are scaled by (1/1-prob). In this
    case, dropout2d performs a channel-wise dropout but assumes a channel is a 2D
    feature map.

    Parameters
    ----------
    x
        a 3D or 4D input array. Should have a floating-point data type.
    prob
        probability of a channel to be zero-ed.
    training
        controls whether dropout2d is performed during training or ignored
        during testing.
    data_format
        "NHWC" or "NCHW". Defaults to "NHWC".
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

    >>> x = ivy.array([[1, 1, 1]])
    >>> y = ivy.dropout2d(x, 0.5)
    >>> print(y)
    ivy.array([[0., 2., 2.]])

    >>> x = ivy.array([[1, 1, 1]])
    >>> y = ivy.dropout2d(x, 1, training=False, data_format="NCW")
    >>> print(y)
    ivy.array([[1, 1, 1]])

    With one :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([[100, 200, 300]]),
                          b=ivy.array([[400, 500, 600]]))
    >>> y = ivy.dropout2d(x, 0.5)
    >>> print(y)
    {
        a: ivy.array([[200., 0., 600.]]),
        b: ivy.array([[0., 0., 1200.]])
    }
    """
    return ivy.current_backend(x).dropout2d(
        x, prob, training=training, data_format=data_format, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def dropout3d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NDHWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Randomly zero out entire channels with probability prob using samples from a
    Bernoulli distribution and the remaining channels are scaled by (1/1-prob). In this
    case, dropout3d performs a channel-wise dropout but assumes a channel is a 1D
    feature map.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def ifft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    *,
    norm: str = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Compute the one dimensional discrete Fourier transform given input at least 1-D
    input x.

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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def embedding(
    weights: Union[ivy.Array, ivy.NativeArray],
    indices: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    max_norm: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Embeds a given tensor of indices using a given tensor of weights.

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
        len(weights.shape), 2, message="weights must be 2-d", as_array=False
    )
    if ivy.exists(out):
        return ivy.inplace_update(
            out,
            ivy.current_backend(indices).embedding(
                weights,
                indices,
                max_norm=max_norm,
                out=out,
            ),
        )
    else:
        return ivy.current_backend(indices).embedding(
            weights,
            indices,
            max_norm=max_norm,
            out=out,
        )


@handle_exceptions
@handle_nestable
@handle_out_argument
@inputs_to_ivy_arrays
def dft(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = 1,
    inverse: bool = False,
    onesided: bool = False,
    dft_length: Optional[Union[int, Tuple[int]]] = None,
    norm: str = "backward",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the discrete Fourier transform of input.

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
        res = ivy.ifft(x, axis, norm=norm, n=dft_length, out=out)
    else:
        res = ivy.fft(x, axis, norm=norm, n=dft_length, out=out)

    if onesided:
        slices = [slice(0, a) for a in res.shape]
        slices[axis] = slice(0, ivy.shape(res, as_array=True)[axis] // 2 + 1)
        res = res[tuple(slices)]
    return res


@handle_exceptions
@handle_nestable
@handle_out_argument
@inputs_to_ivy_arrays
def interp(x, xp, fp, left=None, right=None, period=None):
    x_arr = ivy.array(x)
    fix_later = False
    if x_arr.shape == ():
        x_arr = ivy.array([x])
        fix_later = True
    x = ivy.astype(x_arr, "float64")
    xp = ivy.astype(ivy.array(xp), "float64")
    fp = ivy.astype(ivy.array(fp), "float64")
    ivy.utils.assertions.check_equal(xp.ndim, 1, as_array=False)
    ivy.utils.assertions.check_equal(fp.ndim, 1, as_array=False)
    ivy.utils.assertions.check_equal(xp.shape[0], fp.shape[0], as_array=False)
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


def _tf_area_dim_scale(index, starting_index, scale, ending_index):
    if index < starting_index:
        dim_scale = scale if index + 1 > ending_index else index + 1 - starting_index
    else:
        dim_scale = ending_index - index if index + 1 > ending_index else 1.0
    return dim_scale


def _tf_area_indices(dim_index, scale):
    starting_index = dim_index * scale
    ending_index = (dim_index + 1) * scale
    rounded_indices = (
        int(starting_index),
        math.ceil(ending_index),
    )
    return starting_index, ending_index, rounded_indices


def _tf_area_interpolate(x, size, dims):
    ret = ivy.zeros((x.shape[:2] + size))
    scale = ivy.divide(ivy.shape(x)[2:], size)
    area = 1.0 / ivy.prod(scale)
    for i, ba in enumerate(x):
        for j, ch in enumerate(ba):
            if dims == 3:
                for d_dim in range(size[0]):
                    for h_dim in range(size[1]):
                        for w_dim in range(size[2]):
                            d_in, d_in1, d_index = _tf_area_indices(d_dim, scale[0])
                            h_in, h_in1, h_index = _tf_area_indices(h_dim, scale[1])
                            w_in, w_in1, w_index = _tf_area_indices(w_dim, scale[2])
                            sum_data = ivy.zeros(
                                (
                                    d_index[1] - d_index[0],
                                    h_index[1] - h_index[0],
                                    w_index[1] - w_index[0],
                                )
                            )
                            for d_ind in range(d_index[0], d_index[1]):
                                scale_z = _tf_area_dim_scale(
                                    d_ind, d_in, scale[0], d_in1
                                )
                                for h_ind in range(h_index[0], h_index[1]):
                                    scale_y = _tf_area_dim_scale(
                                        h_ind, h_in, scale[1], h_in1
                                    )
                                    for w_ind in range(w_index[0], w_index[1]):
                                        scale_x = _tf_area_dim_scale(
                                            w_ind, w_in, scale[2], w_in1
                                        )
                                        sum_data[
                                            d_ind - d_index[0],
                                            h_ind - h_index[0],
                                            w_ind - w_index[0],
                                        ] = (
                                            ivy.array(ch[d_ind, h_ind, w_ind])
                                            * scale_x
                                            * scale_y
                                            * scale_z
                                            * area
                                        )
                            ret[i, j, d_dim, h_dim, w_dim] = ivy.sum(sum_data)
            elif dims == 2:
                for h_dim in range(size[0]):
                    for w_dim in range(size[1]):
                        h_in, h_in1, h_index = _tf_area_indices(h_dim, scale[0])
                        w_in, w_in1, w_index = _tf_area_indices(w_dim, scale[1])
                        sum_data = ivy.zeros(
                            (h_index[1] - h_index[0], w_index[1] - w_index[0])
                        )
                        for h_ind in range(h_index[0], h_index[1]):
                            scale_y = _tf_area_dim_scale(h_ind, h_in, scale[0], h_in1)
                            for w_ind in range(w_index[0], w_index[1]):
                                scale_x = _tf_area_dim_scale(
                                    w_ind, w_in, scale[1], w_in1
                                )
                                sum_data[h_ind - h_index[0], w_ind - w_index[0]] = (
                                    ivy.array(ch[h_ind, w_ind])
                                    * scale_x
                                    * scale_y
                                    * area
                                )
                        ret[i, j, h_dim, w_dim] = ivy.sum(sum_data)
            else:
                for w_dim in range(size[0]):
                    w_in, w_in1, w_index = _tf_area_indices(w_dim, scale[0])
                    sum_data = ivy.zeros((w_index[1] - w_index[0],))
                    for w_ind in range(w_index[0], w_index[1]):
                        scale_x = _tf_area_dim_scale(w_ind, w_in, scale[0], w_in1)
                        sum_data[w_ind - w_index[0]] = (
                            ivy.array(ch[w_ind]) * scale_x * area
                        )
                    ret[i, j, w_dim] = ivy.sum(sum_data)
    return ret


def nearest_interpolate(x, dims, size, input_shape, exact):
    off = 0.5 if exact else 0
    for d in range(dims):
        m = input_shape[d + 2]
        n = size[d]
        offsets = (ivy.arange(n, dtype="float32") + off) * m / n
        offsets = ivy.astype(ivy.floor(ivy.astype(offsets, "float32")), "int32")
        x = ivy.gather(x, offsets, axis=d + 2)
    return x


def _triangle_kernel(x):
    return ivy.maximum(0, 1 - ivy.abs(x))


def _cubic_kernel(x):
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = ivy.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return ivy.where(x >= 2.0, 0.0, out)


def _lanczos_kernel(radius, x):
    y = radius * ivy.sin(ivy.pi * x) * ivy.sin(ivy.pi * x / radius)
    out = ivy.where(x != 0, ivy.divide(y, ivy.pi**2 * x**2), 1)
    return ivy.where(ivy.bitwise_and(x >= radius, x < -radius), 0.0, out)


def _dim_scale_factor(input_size, output_size, align_corners, scales):
    if align_corners:
        if output_size > 1:
            dim_scale_factor = (input_size - 1) / (output_size - 1)
        else:
            dim_scale_factor = 0.0
    else:
        dim_scale_factor = (
            input_size / (input_size * scales)
            if scales is not None
            else input_size / output_size
        )
    return dim_scale_factor


def _mitchellcubic_kernel(x):
    absx = abs(x)
    if absx < 1:
        return (7 * absx**3 - 12 * absx**2 + 6) / 6
    elif absx < 2:
        return (-(absx**3) + 6 * absx**2 - 11 * absx + 6) / 6
    else:
        return 0


def _compute_weight_mat(
    input_size,
    output_size,
    scale,
    align_corners,
    kernel_fn,
    antialias: bool,
    dim_scale_factor,
):
    inv_scale = 1.0 / scale
    kernel_scale = ivy.maximum(inv_scale, 1.0) if antialias else 1.0
    if not align_corners:
        sample_f = (ivy.arange(output_size) + 0.5) * dim_scale_factor - 0.5
        x = (
            ivy.abs(
                ivy.expand_dims(sample_f)
                - ivy.expand_dims(ivy.arange(input_size), axis=-1)
            )
            / kernel_scale
        )
    else:
        sample_f = ivy.arange(output_size) * dim_scale_factor
        x = ivy.abs(
            ivy.expand_dims(sample_f) - ivy.expand_dims(ivy.arange(input_size), axis=-1)
        ) / (kernel_scale)
    weights = kernel_fn(x)
    total_weight_sum = ivy.sum(weights, axis=0, keepdims=True)
    weights = ivy.where(
        ivy.abs(total_weight_sum) > 1000.0 * float(ivy.finfo("float32").eps),
        ivy.divide(weights, ivy.where(total_weight_sum != 0, total_weight_sum, 1)),
        0,
    )
    input_size_minus_0_5 = input_size if align_corners else input_size - 0.5
    return ivy.where(
        ivy.expand_dims(
            ivy.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5)
        ),
        weights,
        0,
    )


def _upsample_cubic_convolution1(x, A):
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _upsample_cubic_convolution2(x, A):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _upsample_get_cubic_coefficients(t):
    A = -0.75
    return (
        _upsample_cubic_convolution2(t + 1.0, A),
        _upsample_cubic_convolution1(t, A),
        _upsample_cubic_convolution1(1.0 - t, A),
        _upsample_cubic_convolution2(2.0 - t, A),
    )


def _upsample_cubic_interp1d(coeffs, ts):
    coeffs2 = _upsample_get_cubic_coefficients(ts)
    return _sum_tensors(c1 * c2 for (c1, c2) in zip(coeffs, coeffs2))


def _sum_tensors(ts):
    return _reduce(ivy.add, ts)


def _upsample_bicubic2d_default(
    a,
    output_size,
    align_corners,
    scale_h=None,
    scale_w=None,
):
    N, C, iH, iW = a.shape
    oH, oW = output_size

    def compute_scale(in_size, out_size, align_corners, scale=None):
        if align_corners:
            return (in_size - 1) / (out_size - 1) if out_size > 1 else 0
        else:
            return 1 / scale if scale is not None and scale > 0 else in_size / out_size

    def compute_source_index(scale, dst_index, align_corners):
        if align_corners:
            return scale * dst_index
        else:
            return scale * (dst_index + 0.5) - 0.5

    height_scale = compute_scale(iH, oH, align_corners, scale_h)
    width_scale = compute_scale(iW, oW, align_corners, scale_w)

    N_idx = ivy.reshape(ivy.arange(N), (N, 1, 1, 1))
    C_idx = ivy.reshape(ivy.arange(C), (1, C, 1, 1))
    out_y = ivy.reshape(ivy.arange(oH), ((1, 1, oH, 1)))
    out_x = ivy.reshape(ivy.arange(oW), ((1, 1, 1, oW)))

    real_x = compute_source_index(width_scale, out_x, align_corners)
    in_x = ivy.floor(real_x)
    t_x = real_x - in_x
    ix = ivy.astype(in_x, ivy.int64)

    real_y = compute_source_index(height_scale, out_y, align_corners)
    in_y = ivy.floor(real_y)
    t_y = real_y - in_y
    iy = ivy.astype(in_y, ivy.int64)

    iys_ofs = (iy - 1, iy, iy + 1, iy + 2)
    ixs_ofs = (ix - 1, ix, ix + 1, ix + 2)

    def load_bounded(ys, xs):
        y_idx = ivy.clip(ys, 0, iH - 1)
        x_idx = ivy.clip(xs, 0, iW - 1)
        return a[N_idx, C_idx, y_idx, x_idx]

    def get_x_interp(y):
        coeffs_x = tuple((load_bounded(y, x_ofs) for x_ofs in ixs_ofs))
        return _upsample_cubic_interp1d(coeffs_x, t_x)

    coeffs_y = tuple((get_x_interp(y_ofs) for y_ofs in iys_ofs))
    result = _upsample_cubic_interp1d(coeffs_y, t_y)

    return result


def area_interpolate(x, dims, size, scale):
    ret = ivy.zeros((x.shape[:2] + size))
    inv_scale = ivy.divide(1.0, scale)
    for i, ba in enumerate(x):
        for j, ch in enumerate(ba):
            if dims == 3:
                for d_dim in range(size[0]):
                    for h_dim in range(size[1]):
                        for w_dim in range(size[2]):
                            d_index = (
                                int(d_dim * inv_scale[0]),
                                math.ceil((d_dim + 1) * inv_scale[0]),
                            )
                            h_index = (
                                int(h_dim * inv_scale[1]),
                                math.ceil((h_dim + 1) * inv_scale[1]),
                            )
                            w_index = (
                                int(w_dim * scale[2]),
                                math.ceil((w_dim + 1) * inv_scale[2]),
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
                            int(h_dim * inv_scale[0]),
                            math.ceil((h_dim + 1) * inv_scale[0]),
                        )
                        w_index = (
                            int(w_dim * inv_scale[1]),
                            math.ceil((w_dim + 1) * inv_scale[1]),
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
                        int(w_dim * inv_scale[0]),
                        math.ceil((w_dim + 1) * inv_scale[0]),
                    )
                    scale_x = w_index[1] - w_index[0]
                    ret[i, j, w_dim] = ivy.sum(ch[w_index[0] : w_index[1]]) * (
                        1 / scale_x
                    )
    return ret


def get_interpolate_kernel(mode):
    kernel_func = _triangle_kernel
    if mode == "bicubic_tensorflow":
        kernel_func = lambda inputs: _cubic_kernel(inputs)
    elif mode == "lanczos3":
        kernel_func = lambda inputs: _lanczos_kernel(3, inputs)
    elif mode == "lanczos5":
        kernel_func = lambda inputs: _lanczos_kernel(5, inputs)
    return kernel_func


def generate_einsum_equation(dim):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    input_indices = alphabet[: dim + 2]
    output_indices = [alphabet[2 + i] + alphabet[2 + dim + i] for i in range(dim)]
    contraction_indices = ",".join([input_indices, *output_indices])
    output = input_indices[:2] + "".join([output[-1] for output in output_indices])
    einsum_string = contraction_indices + "->" + output
    return einsum_string


def _interpolate_with_kernel(
    x, dims, size, scale, input_shape, align_corners, antialias, scale_factor, mode
):
    spatial_dims = [2 + i for i in range(dims)]
    equation = generate_einsum_equation(dims)
    kernel_func = get_interpolate_kernel(mode)
    output_shape = tuple(input_shape[:2]) + size
    operands = []
    for i, d in enumerate(spatial_dims):
        m = input_shape[d]
        n = output_shape[d]
        dim_scale_factor = _dim_scale_factor(
            m,
            n,
            align_corners,
            scale_factor[i] if scale_factor is not None else None,
        )
        w = _compute_weight_mat(
            m, n, scale[i], align_corners, kernel_func, antialias, dim_scale_factor
        ).astype(x.dtype)
        operands.append(w)
    return ivy.einsum(equation, x, *operands)


@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@inputs_to_ivy_arrays
@handle_array_function
def interpolate(
    x: Union[ivy.Array, ivy.NativeArray],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "bicubic_tensorflow",
        "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: Optional[bool] = None,
    antialias: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Down/up samples the input to the given size. The algorithm used for interpolation is
    determined by mode.

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
        - nd
        - nearest
        - nearest-exact
        - area
        - tf_area
        - bicubic
        - mitchellcubic
        - lanczos3
        - lanczos5
        - gaussian
    scale_factor
        Multiplier for spatial size that defines the output size (overwriting `size`).
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
    input_shape = ivy.shape(x)
    dims = len(input_shape) - 2
    size = _get_size(scale_factor, size, dims, x.shape)
    if recompute_scale_factor:
        scale_factor = None
    elif scale_factor is not None:
        scale_factor = (
            [scale_factor] * dims
            if isinstance(scale_factor, (int, float))
            else scale_factor
        )
        scale_factor = (
            [scale_factor[0]] * dims
            if isinstance(scale_factor, (list, tuple)) and len(scale_factor) != dims
            else [scale_factor] * dims
        )
    scale = [ivy.divide(size[i], input_shape[i + 2]) for i in range(dims)]
    if mode in [
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "bicubic_tensorflow",
        "lanczos3",
        "lanczos5",
    ]:
        ret = _interpolate_with_kernel(
            x,
            dims,
            size,
            scale,
            input_shape,
            align_corners,
            antialias,
            scale_factor,
            mode,
        )
    elif mode == "bicubic":
        return _upsample_bicubic2d_default(x, size, align_corners)
    elif mode in ["nearest-exact", "nearest"]:
        ret = nearest_interpolate(x, dims, size, input_shape, mode == "nearest-exact")
    elif mode == "area":
        ret = area_interpolate(x, dims, size, scale)
    elif mode == "mitchellcubic":
        batch, channels, in_height, in_width = x.shape
        out_height, out_width = size
        scale_factor_h = out_height / in_height
        scale_factor_w = out_width / in_width
        ret = ivy.zeros((batch, channels, out_height, out_width))
        for i in range(out_height):
            for j in range(out_width):
                p_i = i / scale_factor_h
                p_j = j / scale_factor_w
                left = int(math.floor(p_j - 2))
                right = int(math.ceil(p_j + 2))
                top = int(math.floor(p_i - 2))
                bottom = int(math.ceil(p_i + 2))
                kernel_w = ivy.array(
                    [
                        _mitchellcubic_kernel((p_j - j) / scale_factor_w)
                        for i in range(left, right)
                    ]
                )
                kernel_h = ivy.array(
                    [
                        _mitchellcubic_kernel((p_i - i) / scale_factor_h)
                        for j in range(top, bottom)
                    ]
                )
                left_pad = max(0, -left)
                right_pad = max(0, right - in_width)
                top_pad = max(0, -top)
                bottom_pad = max(0, bottom - in_height)
                pad_width = [(0, 0), (0, 0)] * (len(x.shape) - 3) + [
                    (top_pad, bottom_pad),
                    (left_pad, right_pad),
                ]
                padded_x = ivy.pad(x, pad_width, mode="edge")
                for b in range(batch):
                    for c in range(channels):
                        patch = padded_x[
                            b,
                            c,
                            top + top_pad : bottom + top_pad,
                            left + left_pad : right + left_pad,
                        ]
                        ret[b, c, i, j] = ivy.sum(
                            kernel_h[:, ivy.newaxis] * patch * kernel_w[ivy.newaxis, :]
                        )
    elif mode == "gaussian":
        ratio_h = size[0] / x.shape[-2]
        ratio_w = size[1] / x.shape[-1]
        sigma = max(1 / ratio_h, 1 / ratio_w) * 0.5
        kernel_size = 2 * int(math.ceil(3 * sigma)) + 1
        kernel_h = ivy.zeros((kernel_size,), dtype=x.dtype)
        kernel_w = ivy.zeros((kernel_size,), dtype=x.dtype)
        for i in range(kernel_h.size):
            kernel_h[i] = ivy.exp(-0.5 * ((i - kernel_h.size // 2) / sigma) ** 2)
            kernel_w[i] = ivy.exp(-0.5 * ((i - kernel_w.size // 2) / sigma) ** 2)
        kernel_h /= ivy.sum(kernel_h)
        kernel_w /= ivy.sum(kernel_w)
        pad_width = [(0, 0), (0, 0)] * (len(x.shape) - 3) + [
            (int(math.ceil(3 * sigma)), int(math.ceil(3 * sigma))),
            (int(math.ceil(3 * sigma)), int(math.ceil(3 * sigma))),
        ]
        padded_x = ivy.pad(x, pad_width, mode="constant")
        output_shape = x.shape[:2] + size
        ret = ivy.zeros(output_shape, dtype=x.dtype)
        for i in range(size[0]):
            for j in range(size[1]):
                p_i = int(math.floor(i / ratio_h + int(math.ceil(3 * sigma))))
                p_j = int(math.floor(j / ratio_w + int(math.ceil(3 * sigma))))
                for b in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        patch = padded_x[
                            b,
                            c,
                            p_i - kernel_size // 2 : p_i + kernel_size // 2 + 1,
                            p_j - kernel_size // 2 : p_j + kernel_size // 2 + 1,
                        ]
                        ret[b, c, i, j] = ivy.sum(
                            kernel_h[ivy.newaxis, :] * patch * kernel_w[:, ivy.newaxis]
                        )
    elif mode == "tf_area":
        ret = _tf_area_interpolate(x, size, dims)
    return ivy.astype(ret, ivy.dtype(x), out=out)


interpolate.mixed_backend_wrappers = {
    "to_add": ("handle_device_shifting",),
    "to_skip": (),
}


def _get_size(scale_factor, size, dims, x_shape):
    if scale_factor is not None:
        if isinstance(scale_factor, (float, int)):
            scale_factor = [scale_factor] * dims
        elif isinstance(scale_factor, (tuple, list)) and len(scale_factor) != dims:
            scale_factor = [scale_factor[0]] * dims

        size = tuple(
            [int(math.floor(x_shape[2 + i] * scale_factor[i])) for i in range(dims)]
        )
    else:
        size = (size,) * dims if isinstance(size, int) else tuple(size)
    return size


def _output_ceil_shape(w, f, p, s):
    return math.ceil((w - f + p) / s) + 1


def _padding_ceil_mode(w, f, p, s, return_added_padding=False):
    remaining_pixels = (w - f + sum(p)) % s
    added_padding = 0
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
        added_padding = new_pad - sum(p)
        p = (
            p[0],
            p[1] + added_padding,
        )
    if return_added_padding:
        return p, added_padding
    return p


interpolate.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


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


@handle_nestable
@inputs_to_ivy_arrays
def adaptive_avg_pool1d(
    input: Union[ivy.Array, ivy.NativeArray],
    output_size: int,
) -> ivy.Array:
    """
    Apply a 1D adaptive average pooling over an input signal composed of several input
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
    if input.ndim == 2:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif input.ndim != 3:
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
        ret = ivy.mean(vals, axis=-1)
        ret = ivy.squeeze(ret, axis=0) if squeeze else ret
        return ret

    vals, length_w = _mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i in range(vals.shape[-1]):
        if ret is None:
            ret = vals[..., i]
        else:
            ret = ret + vals[..., i]
    pooled_output = ret / length_w.astype(ret.dtype)

    pooled_output = ivy.squeeze(pooled_output, axis=0) if squeeze else pooled_output
    return pooled_output


adaptive_avg_pool1d.mixed_backend_wrappers = {
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
@inputs_to_ivy_arrays
@handle_array_function
def adaptive_avg_pool2d(
    input: Union[ivy.Array, ivy.NativeArray],
    output_size: Union[Sequence[int], int],
) -> ivy.Array:
    """
    Apply a 2D adaptive average pooling over an input signal composed of several input
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
    if input.ndim == 3:
        input = ivy.expand_dims(input, axis=0)
        squeeze = True
    elif input.ndim != 4:
        raise ivy.utils.exceptions.IvyException(
            f"Got {len(input.shape)}D input, but only 3D and 4D inputs are supported.",
        )

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

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
        ret = ivy.mean(vals, axis=(-3, -1))
        ret = ivy.squeeze(ret, axis=0) if squeeze else ret
        return ret

    vals, length_h = _mask(vals, length_h, range_max_h, dim=-2)
    vals, length_w = _mask(vals, length_w, range_max_w, dim=-1)

    ret = None
    for i, j in itertools.product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    pooled_output = ret / (length_h * length_w).astype(vals.dtype)

    pooled_output = ivy.squeeze(pooled_output, axis=0) if squeeze else pooled_output
    return pooled_output


adaptive_avg_pool2d.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


def _conv_view(lhs, rhs_shape, window_strides, pads, pad_value):
    def _pad(arr, pads, pad_value):
        out = ivy.astype(
            ivy.pad(
                arr,
                ivy.maximum(0, pads).to_list(),
                mode="constant",
                constant_values=ivy.to_scalar(pad_value),
            ),
            arr.dtype,
        )
        slices = tuple(
            _slice(abs(lo) if lo < 0 else 0, hi % dim if hi < 0 else None)
            for (lo, hi), dim in zip(pads, arr.shape)
        )
        return out[slices]

    if (
        _min(lhs.ndim, len(rhs_shape)) < 2
        or lhs.ndim != len(rhs_shape)
        or lhs.shape[1] != rhs_shape[1]
    ):
        raise ValueError("Dimension mismatch")
    if len(window_strides) != len(rhs_shape) - 2:
        raise ValueError("Wrong number of strides for spatial dimensions")
    if len(pads) != len(rhs_shape) - 2:
        raise ValueError("Wrong number of pads for spatial dimensions")

    lhs = _pad(lhs, [(0, 0)] * 2 + list(pads), pad_value)
    in_shape = lhs.shape[2:]
    filter_shape = rhs_shape[2:]
    dim = len(filter_shape)

    out_strides = ivy.multiply(window_strides, lhs.strides[2:]).to_list()
    view_strides = lhs.strides[:1] + tuple(out_strides) + lhs.strides[1:]

    out_shape = [
        (in_shape[i] - filter_shape[i]) // s + 1 for i, s in enumerate(window_strides)
    ]
    view_shape = list(lhs.shape[:1]) + out_shape + rhs_shape[1:]

    view = ivy.as_strided(lhs, view_shape, view_strides)

    view_axes = list(range(view.ndim))
    sum_axes = view_axes[-dim - 1 :]
    rhs_axes = [view.ndim] + sum_axes
    out_axes = [0, view.ndim] + list(range(1, dim + 1))

    return view, view_axes, rhs_axes, out_axes


def _dilate(operand, factors, fill_value):
    outspace = list(operand.shape[:2]) + [
        shape + (factors[i] - 1) * (shape - 1)
        for i, shape in enumerate(operand.shape[2:])
    ]
    out = ivy.full(
        outspace,
        ivy.to_scalar(fill_value),
        dtype=fill_value.dtype,
    )
    lhs_slices = tuple(_slice(None, None, step) for step in factors)
    out[(_slice(None),) * 2 + lhs_slices] = operand
    return out


def _padtype_to_pads(in_shape, filter_shape, window_strides, padding):
    if padding.upper() == "SAME":
        out_shape = [
            math.ceil(in_size / stride)
            for in_size, stride in zip(in_shape, window_strides)
        ]
        pad_sizes = [
            _max((out_size - 1) * stride + filter_size - in_size, 0)
            for out_size, stride, filter_size, in_size in zip(
                out_shape, window_strides, filter_shape, in_shape
            )
        ]
        return [(pad_size // 2, pad_size - pad_size // 2) for pad_size in pad_sizes]
    else:
        return [(0, 0)] * len(in_shape)


identities = {
    "max": -float("inf"),
    "min": float("inf"),
    "add": 0,
    "mul": 1,
    "multiply": 1,
    "logical_and": True,
    "logical_or": False,
}


def _cast_init(init, dtype):
    if not ivy.is_bool_dtype(dtype) and ivy.isinf(init):
        if ivy.is_float_dtype(dtype):
            info = ivy.finfo(dtype)
        else:
            info = ivy.iinfo(dtype)
        if "float64" not in str(dtype):
            init = info.max if init > 0 else info.min
    return ivy.array(init, dtype=dtype)


def _get_identity(func, dtype, init):
    func_name = func.__name__
    if func_name in identities:
        identity = identities[func_name]
        return _cast_init(identity, dtype)
    return init


avg_pool2d.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def reduce_window(
    operand: Union[ivy.Array, ivy.NativeArray],
    init_value: Union[int, float],
    computation: Callable,
    window_dimensions: Union[int, Sequence[int]],
    /,
    *,
    window_strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Tuple[int, int]]] = "VALID",
    base_dilation: Union[int, Sequence[int]] = 1,
    window_dilation: Union[int, Sequence[int]] = 1,
) -> ivy.Array:
    """
    Apply a reduction function to all elements in each window of an array.

    Parameters
    ----------
    operand
        An array representing the base area on which the window is going to slide over.
    init_value
        The starting value for the reduction.
    computation
        The reduction function to apply to elements in each window.
    window_dimensions
        A sequence containing the window dimensions.
    window_strides
        A sequence containing the window strides.
    padding
        Either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    base_dilation
        A sequence containing the base dilation values.
    window_dilation
        A sequence containing the window dilation values.

    Returns
    -------
    ret
        The result of the pooling-like operation.

    Examples
    --------
    >>> x = ivy.array([[1, 2, 3, 4],
    >>>                [5, 6, 7, 8],
    >>>                [9, 10, 11, 12]])
    >>> ivy.reduce_window(x, 0, ivy.add, (2, 2))
    ivy.array([[14, 18, 22], [30, 34, 38]])
    """
    # ToDo: add support for window_dilation
    computation = _correct_ivy_callable(computation)
    op = operand
    dims, strides, padding, base_dilation, window_dilation = map(
        lambda x: tuple([x] * len(op.shape)) if isinstance(x, int) else x,
        [window_dimensions, window_strides, padding, base_dilation, window_dilation],
    )
    init_value = _cast_init(init_value, op.dtype)
    identity = _get_identity(computation, operand.dtype, init_value)
    if isinstance(padding, str):
        pads = _padtype_to_pads(op.shape, dims, strides, padding)
    else:
        pads = padding
    op = op.reshape((1, 1) + op.shape)
    if base_dilation:
        op = _dilate(op, base_dilation, identity)
    view = _conv_view(op, [1, 1] + list(dims), strides, pads, identity)[0]
    view = ivy.reshape(view, (*view.shape[1 : 1 + len(dims)], -1))
    ret = ivy.reduce(view, init_value, computation, axes=-1)
    return ret.astype(operand.dtype)


reduce_window.mixed_backend_wrappers = {
    "to_add": (
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
# @outputs_to_ivy_arrays
def fft2(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    s: Sequence[int] = None,
    dim: Sequence[int] = (-2, -1),
    norm: str = "backward",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Compute the 2-dimensional discrete Fourier Transform.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*,
        where d_in indicates the dimension that needs FFT2.
    s
        sequence of ints, optional
        Shape (length of each transformed axis) of the output (s[0] refers to axis 0,
        s[1] to axis 1, etc.). This corresponds to n for fft(x, n). Along each axis,
        if the given shape is smaller than that of the input, the input is cropped.
        If it is larger, the input is padded with zeros. if s is not given, the shape
        of the input along the axes specified by axes is used.
    dim
        Axes over which to compute the FFT2. If not given, the last two axes are used.
        A repeated index in axes means the transform over that axis is performed
        multiple times. A one-element sequence means that a one-dimensional FFT is
        performed.
    norm
        Optional argument, "backward", "ortho" or "forward". Defaults to be "backward".
        "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$.
        "forward" indicates normalization by $\frac{1}{n}$.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the FFT2 operation.

    Examples
    --------
    >>> a = ivy.array([[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4]])
    >>> ivy.fft2(a)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
             0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
             0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
             0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
             0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])
    """
    return ivy.current_backend(x).fft2(x, s=s, dim=dim, norm=norm, out=out)


fft2.mixed_backend_wrappers = {
    "to_add": ("handle_device_shifting",),
    "to_skip": (),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def ifftn(
    x: Union[ivy.Array, ivy.NativeArray],
    s: Optional[Union[int, Tuple[int, ...]]] = None,
    axes: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    norm: str = "backward",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Compute the N-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    x
        Input array of complex numbers.
    s
        Shape (length of transformed axis) of the output (`s[0]` refers to axis 0,
        `s[1]` to axis 1, etc.). If given shape is smaller than that of the input,
        the input is cropped. If larger, input is padded with zeros. If `s` is not
        given, shape of input along axes specified by axes is used.
    axes
        Axes over which to compute the IFFT. If not given, last `len(s)` axes are
        used, or all axes if `s` is also not specified. Repeated indices in axes
        means inverse transform over that axis is performed multiple times.
    norm
        Indicates direction of the forward/backward pair of transforms is scaled
        and with what normalization factor. "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$. "forward"
        indicates normalization by $\frac{1}{n}$.
    out
        Optional output array for writing the result to. It must have a shape that
        the inputs broadcast to.

    Returns
    -------
    out
        The truncated or zero-padded input, transformed along the axes indicated
        by axes, or by a combination of s or x, as explained in the parameters
        section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of axes is larger than the number of axes of x.

    Examples
    --------
    >>> x = ivy.array([[0.24730653+0.90832391j, 0.49495562+0.9039565j,
                        0.98193269+0.49560517j],
                        [0.93280757+0.48075343j, 0.28526384+0.3351205j,
                        0.2343787 +0.83528011j],
                        [0.18791352+0.30690572j, 0.82115787+0.96195183j,
                        0.44719226+0.72654048j]])
    >>> y = ivy.ifftn(x)
    >>> print(y)
    ivy.array([[ 0.51476765+0.66160417j, -0.04319742-0.05411636j,
            -0.015561  -0.04216015j],
            [ 0.06310689+0.05347854j, -0.13392983+0.16052352j,
            -0.08371392+0.17252843j],
            [-0.0031429 +0.05421245j, -0.10446617-0.17747098j,
            0.05344324+0.07972424j]])

    >>> b = ivy.ifftn(x, s=[2, 1], axes=[0, 1], norm='ortho')
    >>> print(b)
    ivy.array([[ 0.8344667 +0.98222595j],
            [-0.48472244+0.30233797j]])
    """
    return ivy.current_backend(x).ifftn(x, s=s, axes=axes, norm=norm, out=out)


@handle_exceptions
@handle_nestable
@handle_out_argument
# @inputs_to_ivy_arrays
@to_native_arrays_and_back
def rfftn(
    x: Union[ivy.Array, ivy.NativeArray],
    s: Optional[Sequence[int]] = None,
    axes: Optional[Sequence[int]] = None,
    *,
    norm: Optional[str] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the N-dimensional discrete Fourier Transform for real input.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (s[0] refers to axis 0, s[1] to axis 1, etc.). The final element of s
        corresponds to n for rfft(x, n), while for the remaining axes, it
        corresponds to n for fft(x, n). Along any axis, if the given shape is
        smaller than that of the input, the input is cropped. If it is larger,
        the input is padded with zeros. If s is not given, the shape of the
        input along the axes specified by axes is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last len(s) axes
        are used, or all axes if s is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode. Default is "backward". Indicates which direction of
        the forward/backward pair of transforms is scaled and with what
        normalization factor.
    out : array_like, optional
        Optional output array to store the result of the computation. The shape
        and dtype of this array must match the expected output.

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes indicated
        by axes or by a combination of s and a, as explained in the parameters
        section above. The length of the last axis transformed will be
        s[-1] // 2 + 1, while the remaining transformed axes will have lengths
        according to s, or unchanged from the input.

    Raises
    ------
    ValueError
        If s and axes have different lengths.
    IndexError
        If an element of axes is larger than the number of axes of a.

    Examples
    --------
    >>> x = ivy.array([1, 2, 3, 4], dtype=ivy.float32)
    >>> result = ivy.rfftn(x)
    >>> print(result)
    [10.+0.j  -2.+2.j   0.+0.j  -2.-2.j]

    >>> x = ivy.array([[1, 2, 3], [4, 5, 6]], dtype=ivy.float32)
    >>> result = ivy.rfftn(x, s=(3, 4), axes=(0, 1))
    >>> print(result)
    [[21. +0.j    0. +0.j    0. +0.j    0. +0.j   ]
     [-1.5+1.299j -1.5+0.433j -1.5-0.433j -1.5-1.299j]]
    """
    if norm is None:
        norm = "backward"

    if axes is None:
        axes = list(range(x.ndim - len(s), x.ndim))
    elif s is None:
        s = [x.shape[axis] for axis in axes]
    elif len(s) != len(axes):
        raise ValueError("s and axes must have the same length.")

    return ivy.current_backend(x).rfftn(x, s=s, axes=axes, norm=norm, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
def irfftn(
    x: Union[ivy.Array, ivy.NativeArray],
    s: Optional[Union[int, Tuple[int, ...]]] = None,
    axes: Optional[Union[int, Tuple[int, ...]]] = None,
    *,
    norm: str = "backward",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""
    Compute the inverse of `rfftn`.

    This function computes the inverse of the N-dimensional discrete Fourier Transform
    for real input over any number of axes in an M-dimensional array by means of the
    Fast Fourier Transform (FFT). In other words, `irfftn(rfftn(x), x.shape) == x` to
    within numerical accuracy. (The `x.shape` is necessary like `len(x)` is for irfft,
    and for the same reason.)

    The input should be ordered in the same way as is returned by rfftn.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output (`s[0]` refers to axis 0,
        `s[1]` to axis 1, etc.). `s` is also the number of input points used along this
        axis, except for the last axis, where `s[-1]//2+1` points of the input are used.
        Along any axis, if the shape indicated by s is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros. If s is
        not given, the shape of the input along the axes specified by axes is used.
        Except for the last axis which is taken to be 2*(m-1) where m is the length of
        the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT. If not given, last `len(s)` axes are
        used, or all axes if `s` is also not specified. Repeated indices in axes
        means inverse transform over that axis is performed multiple times.
    norm : str, optional
        Indicates direction of the forward/backward pair of transforms is scaled
        and with what normalization factor. "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$. "forward"
        indicates normalization by $\frac{1}{n}$.
    out
        Optional output array for writing the result to. It must have a shape that
        the inputs broadcast to.

    Returns
    -------
    out
        The truncated or zero-padded input, transformed along the axes indicated by axes
        or by a combination of s or a, as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding element of
        s, or the length of the input in every axis except for the last one if s is not
        given. In the final transformed axis the length of the output when s is not
        given is `2*(m-1)` where `m` is the length of the final transformed axis of the
        input. To get an odd number of output points in the final axis, s must be
        specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of axes is larger than the number of axes of x.

    Examples
    --------
    >>> x = ivy.zeros((3, 2, 2))
    >>> x[0, 0, 0] = 3 * 2 * 2
    >>> result = ivy.irfftn(x)
    >>> print(result)
    ([[[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]],
       [[1.,  1.],
        [1.,  1.]]])
    """
    return ivy.current_backend(x).irfftn(x, s=s, axes=axes, norm=norm, out=out)
