# global
from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
    inputs_to_ivy_arrays,
    handle_device_shifting,
    handle_backend_invalid,
)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the logit of x.

    logit(x) = log(x / (1 - x)).

    Parameters
    ----------
    x
        Input data.
    eps
        When eps is None the function outpus NaN where x < 0 or x > 1.
        and inf or -inf where x = 1 or x = 0, respectively.
        Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
    out
        Optional output array.

    Returns
    -------
    ret
        Array containing elementwise logits of x.

    Examples
    --------
    >>> x = ivy.array([1, 0, 0.9])
    >>> z = ivy.logit(x)
    >>> print(z)
    ivy.array([       inf,       -inf, 2.19722438])

    >>> x = ivy.array([1, 2, -0.9])
    >>> z = ivy.logit(x, eps=0.2)
    >>> print(z)
    ivy.array([ 1.38629448,  1.38629448, -1.38629436])
    """
    return current_backend(x).logit(x, eps=eps, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_ivy_arrays
def prelu(
    x: Union[ivy.NativeArray, ivy.Array],
    slope: Union[float, ivy.NativeArray, ivy.Array],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Prelu takes input data (Array) and slope array as input,

    and produces one output data (array) where the function
    f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
    to the data array elementwise. This operator supports unidirectional
    broadcasting (array slope should be unidirectional broadcastable to
    input tensor X);

    Parameters
    ----------
    x
        Input Array.
    slope
        Slope Array. The shape of slope can be smaller then first input X;
        if so, its shape must be unidirectional broadcastable to X.
    out
        Optional output array.

    Returns
    -------
    ret
         Array containing Parametrized relu values.
    """
    try:
        return ivy.where(x > 0, x, x * slope, out=out)
    except ivy.utils.exceptions.IvyError(
        f"The shape {slope.shape} is not Unidirectional Broadcastable\n"
        "as per ONNX standards"
    ) as IvyException:
        if len(slope.shape) == 1:
            dim = slope.shape[0]
            new_shape = []
            n = 0
            for d in x.shape:
                if d == dim:
                    new_shape.append(d)
                    n += 1
                else:
                    new_shape.append(d)
            if n == 1:
                xs = x * slope.reshape(tuple(new_shape), out=out)
                return ivy.where(x > 0, x, xs, out=out)
        raise IvyException


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def thresholded_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the rectified linear unit function with custom threshold.

    Parameters
    ----------
    x
        input array
    threshold
        threshold value above which the activation is linear. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit activation of each element in
        ``x``. with custom ``threshold``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.thresholded_relu(x, threshold=0.5)
    >>> print(y)
    ivy.array([0.,  0. ,  1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.thresholded_relu(x, threshold=1, out = y)
    >>> print(y)
    ivy.array([ 1.5,  0., 0.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.2, 0.6]))
    >>> x = ivy.thresholded_relu(x, threshold=0.5)
    >>> print(x)
    {
        a: ivy.array([1., 0.]),
        b: ivy.array([0., 0.6])
    }
    """
    return current_backend(x).thresholded_relu(x, threshold=threshold, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def relu6(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the rectified linear unit 6 function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit 6 activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.relu6(x)
    >>> print(y)
    ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.zeros(9)
    >>> ivy.relu6(x, out = y)
    >>> print(y)
    ivy.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])
    """
    return current_backend(x).relu6(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def logsigmoid(
    input: Union[ivy.NativeArray, ivy.Array], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply element-wise Log-sigmoid of x.

    logsigmoid(x) = log(1 / (1 + exp(-x)).

    Parameters
    ----------
    input
        Input array.

    Returns
    -------
        Array with same shape as input with Log-sigmoid applied to every element.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> z = x.logsigmoid()
    >>> print(z)
    ivy.array([-1.31326175, -0.69314718, -0.31326169])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> z = x.logsigmoid()
    >>> print(z)
    ivy.array([-0.20141329, -0.40318608, -2.48683619])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.2, 0.6]))
    >>> x = ivy.logsigmoid(x)
    >>> print(x)
    {
        a: ivy.array([-0.31326169, -1.46328247]),
        b: ivy.array([-0.59813893, -0.43748799])
    }
    """
    return ivy.current_backend(input).logsigmoid(input, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def selu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the scaled exponential linear unit function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the scaled exponential linear unit activation of each
        element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.selu(x)
    >>> print(y)
    ivy.array([-1.11133075,  0.        ,  1.05070102,  2.10140204,  3.15210295,
            4.20280409,  5.25350523,  6.30420589,  7.35490704])
    >>> x = ivy.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = ivy.zeros(9)
    >>> ivy.selu(x, out = y)
    >>> print(y)
    ivy.array([-1.11133075,  0.        ,  1.05070102,  2.10140204,  3.15210295,
            4.20280409,  5.25350523,  6.30420589,  7.35490704])

    With :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
    ...                   b=ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    ...                   )
    >>> x = ivy.selu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([-1.6705687, -1.52016652, -1.11133075, 0., 1.05070102,
                      2.10140204, 3.15210295, 4.20280409, 5.25350523]),
        b: ivy.array([1.05070102, 2.10140204, 3.15210295, 4.20280409, 5.25350523,
                      6.30420589, 7.35490704, 8.40560818, 9.45630932])
    }
    """
    return current_backend(x).selu(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def silu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the silu function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the silu activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.silu(x)
    >>> print(y)
    ivy.array([-0.2689,  0.7310,  1.7615])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.silu()
    >>> print(y)
    ivy.array([-0.2689,  0.7310,  1.7615])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.silu(x)
    >>> print(y)
    ivy.array([[-0.2784,  3.7168,  1.8708], [ 1.4374,  4.1379, -0.0089]])
    """
    return current_backend(x).silu(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def elu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the elu unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        scaler for controlling the slope of the function for x <= 0 Default: 1.0
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with elu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.elu(x)
    >>> print(y)
    ivy.array([ 0.38999999, -0.57258511])
    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.elu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5, 0.69999999, -0.90928203])
    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.elu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.10000002,  2.20000005,  3.29999995],
           [-0.98772264, -0.99591321, -0.99863964]])
    With :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.elu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.69880581]),
        b: ivy.array([0.40000001, -0.18126924])
    }
    """
    return current_backend(x).elu(x, alpha=alpha, out=out)


def sequence_length(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.int64:
    """
    Produce a scalar (tensor of empty shape) containing the number of tensors in the ivy
    array input.

    Parameters
    ----------
    x
        Can be a sequence of any tensor type: bool, complex128,
        complex64, double, float, float16, int16, int32, int64,
        int8, string, uint16, uint32, uint64, uint8

    Returns
    -------
    length
        Length of the input sequence, as a scalar (empty shape tensor).

    Examples
    --------
    >>> x = ivy.array([True, False, True])
    >>> y = ivy.sequence_length(x)
    >>> print(y)
    3

    >>> x = [1.0, 2.5, -3.4, 5.6, -85.3]
    >>> y = ivy.sequence_length(x)
    >>> print(y)
    5
    """
    return current_backend(x).sequence_length(x, out=out)
