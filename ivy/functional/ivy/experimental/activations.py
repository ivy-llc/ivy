# global
from typing import Union, Optional, Callable, Literal

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
    handle_device,
    handle_backend_invalid,
    handle_complex_input,
)


def _logit_jax_like(
    x: Union[float, int, ivy.Array],
    /,
    *,
    fn_original: Optional[Callable] = None,
    eps: Optional[float] = None,
    out: Optional[ivy.Array] = None,
):
    real = ivy.real(x)
    imag = ivy.imag(x)
    if eps is None:
        real = ivy.where(ivy.logical_or(real > 1, real < 0), ivy.nan, real)
    else:
        real = ivy.clip(real, eps, 1 - eps)
    z = ivy.add(real, ivy.multiply(ivy.array(1j, dtype=x.dtype), imag))
    z = ivy.log(z / (1 - z))
    return z


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
@handle_complex_input
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the logit of x.

    logit(x) = log(x / (1 - x)).

    Parameters
    ----------
    x
        Input data.
    eps
        When eps is None the function outputs NaN where x < 0 or x > 1.
        and inf or -inf where x = 1 or x = 0, respectively.
        Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
    complex_mode
        optional specifier for how to handle complex data types. See
        ``ivy.func_wrapper.handle_complex_input`` for more detail.
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


logit.jax_like = _logit_jax_like


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
    """Prelu takes input data (Array) and slope array as input,

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
                    n += 1
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
@handle_device
def thresholded_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the rectified linear unit function with custom threshold.

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


def _relu6_jax_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    fn_original=None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(
        ivy.logical_or(
            ivy.real(x) < 0, ivy.logical_and(ivy.real(x) == 0, ivy.imag(x) < 0)
        ),
        ivy.array(0, dtype=x.dtype),
        ivy.where(
            ivy.logical_or(
                ivy.real(x) > 6, ivy.logical_and(ivy.real(x) == 6, ivy.imag(x) > 0)
            ),
            ivy.array(6, dtype=x.dtype),
            x,
        ),
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def relu6(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the rectified linear unit 6 function element-wise.

    Parameters
    ----------
    x
        input array
    complex_mode
        optional specifier for how to handle complex data types. See
        ``ivy.func_wrapper.handle_complex_input`` for more detail.
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


relu6.jax_like = _relu6_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
@handle_complex_input
def logsigmoid(
    input: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply element-wise Log-sigmoid of x.

    logsigmoid(x) = log(1 / (1 + exp(-x)).

    Parameters
    ----------
    input
        Input array.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``ivy.func_wrapper.handle_complex_input`` for more detail.

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
@handle_device
def selu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Apply the scaled exponential linear unit function element-wise.

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
@handle_device
def silu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Apply the silu function element-wise.

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
    """Apply the elu unit function element-wise.

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


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def hardtanh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    max_val: float = 1,
    min_val: float = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the hardtanh unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    min_val
        minimum value of the linear region range. Default: -1.
    max_val
        maximum value of the linear region range. Default: 1.
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
    >>> y = ivy.hardtanh(x)
    >>> print(y)
    ivy.array([ 0.39, -0.85])
    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.hardtanh(x, out=y)
    >>> print(y)
    ivy.array([ 1., 0.7, -1.])
    >>> x = ivy.array([[1.1, 2.2, 3.3],[-0.4, 0.5, -6.6]])
    >>> ivy.hardtanh(x, out=x)
    >>> print(x)
    ivy.array([[ 1.,  1., 1.],[-0.4, 0.5, -1.]])

    With :class:`ivy.Container` input:
    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.hardtanh(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -1.]),
        b: ivy.array([0.4, -0.2])
    }
    """
    return current_backend(x).hardtanh(x, max_val=max_val, min_val=min_val, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def tanhshrink(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Apply the tanhshrink function element-wise.

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
        an array containing the tanhshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.tanhshrink(x)
    >>> print(y)
    ivy.array([-0.23840582,  0.23840582,  1.03597236])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.tanhshrink()
    >>> print(y)
    ivy.array([-0.23840582,  0.23840582,  1.03597236])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.tanhshrink(x)
    >>> print(y)
    ivy.array([[-0.43827677,  2.80100036,  1.12954807],
                [ 0.76459098,  3.20044947, -5.60000372]])
    """
    return current_backend(x).tanhshrink(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def softshrink(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    lambd: float = 0.5,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the softshrink function element-wise.

    Parameters
    ----------
    x
        input array.
    lambd
        the value of the lower bound of the linear region range.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         an array containing the softshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.softshrink(x)
    >>> print(y)
    ivy.array([-0.5,  0.5,  1.5])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.softshrink()
    >>> print(y)
    ivy.array([-0.5,  0.5,  1.5])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.softshrink(x)
    >>> print(y)
    ivy.array([[-0.79999995,  3.29999995,  1.59999991],
       [ 1.20000005,  3.69999981, -6.0999999 ]])
    """
    return current_backend(x).softshrink(x, lambd=lambd, out=out)


def _celu_jax_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    fn_original: Optional[Callable] = None,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    # implementation of max(0, x) for complex numbers
    complex_max = ivy.where(
        (
            ivy.logical_or(
                ivy.real(x) < 0, ivy.logical_and(ivy.real(x) == 0, ivy.imag(x) < 0)
            )
        ),
        ivy.astype(0.0, x.dtype),
        x,
    )

    # implementation of min(0, x) for complex numbers
    complex_min = ivy.where(
        (
            ivy.logical_or(
                ivy.real(x) < 0, ivy.logical_and(ivy.real(x) == 0, ivy.imag(x) < 0)
            )
        ),
        x,
        ivy.astype(0.0, x.dtype),
    )
    return complex_max + alpha * ivy.expm1(complex_min / alpha)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def threshold(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: float,
    value: float,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the threshold function element-wise.

    Parameters
    ----------
    x
        input array.
    threshold
        The value to threshold at.
    value
        The value to replace with.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the threshold activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.threshold(x,value=0.0, threshold=1.5)
    >>> print(y)
    ivy.array([0., 0., 2.])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> x.threshold(value=0.0, threshold=1.5)
    >>> print(y)
    ivy.array([0., 0., 2.])


    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.threshold(x, value=0.0, threshold=1.5)
    >>> print(y)
    ivy.array([[0.        , 3.79999995, 2.0999999 ],
            [1.70000005, 4.19999981, 0.        ]])
    """
    return current_backend(x).threshold(x, threshold=threshold, value=value, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def celu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the Continuously Differentiable Exponential Linear Unit (CELU)
    activation function to each element of the input.

    Parameters
    ----------
    x
        Input array.
    alpha
        The alpha value (negative slope) for the CELU formulation. Default is ``1.0``
    complex_mode
        optional specifier for how to handle complex data types. See
        ``ivy.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with celu applied element-wise.


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.celu(x)
    >>> y
    ivy.array([ 0.39, -0.57])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.39, -0.85]), b=ivy.array([1., -0.2]))
    >>> y = ivy.celu(x)
    >>> y
    {
        a: ivy.array([0.38999999, -0.57]),
        b: ivy.array([1., -0.18])
    }
    """
    return current_backend(x).celu(x, alpha=alpha, out=out)


celu.jax_like = _celu_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def scaled_tanh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the scaled hyperbolic tangent (tanh) activation.

    The scaled tanh activation function is defined as:
    out = alpha * tanh(beta * x)


    Parameters
    ----------
    x
        input array.
    alpha
        The scaling parameter for the output.
        Determines the amplitude of the tanh function.
        Default: 1.7159
    beta
        The scaling parameter for the input.
        Determines the slope of the tanh function.
        Default: 0.67
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         The input array after applying the scaled tanh activation.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([22.])
    >>> y = ivy.scaled_tanh(x)
    >>> y
    ivy.array([1.71589994]))

    >>> x = ivy.array([4.0, 7.0])
    >>> y = ivy.scaled_tanh(x, alpha=1.2, beta=5)
    >>> y
    ivy.array([1.20000005, 1.20000005])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.2, -1.2]), b=ivy.array([4.4, -2.2]))
    >>> y = ivy.scaled_tanh(x)
    >>> y
    {
        a: ivy.array([1.14324772, -1.14324772]),
        b: ivy.array([1.70648694, -1.54488957])
    }
    >>> x = ivy.Container(a=ivy.array([1.2]), b=ivy.array([4.4]))
    >>> y = ivy.scaled_tanh(x, alpha=0.2, beta=0.5)
    >>> y
    {
    a: ivy.array([0.10740992]),
    b: ivy.array([0.19514863])
    }
    """
    return current_backend(x).scaled_tanh(x, alpha=alpha, beta=beta, out=out)


stanh = scaled_tanh


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def hardshrink(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    lambd: float = 0.5,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply the hardshrink function element-wise.

    Parameters
    ----------
    x
        input array.
    lambd
        the value for the Hardshrink formulation.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         an array containing the hardshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.hardshrink(x)
    >>> print(y)
    ivy.array([-1.,  1.,  2.])
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.hardshrink()
    >>> print(y)
    ivy.array([-1.,  1.,  2.])
    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.hardshrink(x)
    >>> print(y)
    ivy.array([[-1.29999995,  3.79999995,  2.0999999 ],
       [ 1.70000005,  4.19999981, -6.5999999 ]])
    """
    return current_backend(x).hardshrink(x, lambd=lambd, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def hardsilu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Apply the hardsilu/hardswish function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
        an array containing the output of the hardsilu/hardswish function applied
        to each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.hardsilu(x)
    >>> print(y)
    ivy.array([0.66666669, 1.66666663, 3.        ])
    >>> x = ivy.array([-2.1241, 1.4897, 4.4090])
    >>> y = ivy.zeros(3)
    >>> ivy.hardsilu(x, out=y)
    >>> print(y)
    ivy.array([-0.31008321,  1.1147176 ,  4.40899992])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-0.5, -1, 0]), b=ivy.array([0.5, 1., 2]))
    >>> y = ivy.hardsilu(x)
    >>> print(y)
    {
        a: ivy.array([-0.20833333, -0.33333334, 0.]),
        b: ivy.array([0.29166666, 0.66666669, 1.66666663])
    }
    """
    return current_backend(x).hardsilu(x, out=out)
