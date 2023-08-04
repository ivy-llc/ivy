"""Collection of Ivy activation functions."""

from typing import Union, Optional, Callable, Literal

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device_shifting,
    handle_complex_input,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def gelu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    approximate: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the Gaussian error linear unit (GELU) activation function.

    Parameters
    ----------
    x
        Input array.
    approximate
        Whether to approximate, default is ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with gelu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.2, -0.6, 1.5])
    >>> y = ivy.gelu(x)
    >>> y
    ivy.array([-0.138, -0.165, 1.4])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([-1.3, 3.8, 2.1])
    >>> y = ivy.gelu(x)
    >>> y
    ivy.array([-0.126, 3.8, 2.06])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1., 2.]), b=ivy.array([-0.9, -1.]))
    >>> y = ivy.gelu(x)
    >>> y
    {
        a: ivy.array([0.841, 1.95]),
        b: ivy.array([-0.166, -0.159])
    }
    """
    return current_backend(x).gelu(x, approximate=approximate, out=out)


def _leaky_relu_jax_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    fn_original: Optional[Callable] = None,
    alpha: float = 0.2,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.where(
        (
            ivy.logical_or(
                ivy.real(x) < 0, ivy.logical_and(ivy.real(x) == 0, ivy.imag(x) < 0)
            )
        ),
        ivy.astype(x * alpha, x.dtype),
        x,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
@handle_complex_input
def leaky_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 0.2,
    out: Optional[ivy.Array] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
) -> ivy.Array:
    """
    Apply the leaky rectified linear unit function element-wise.

    If the input is complex, then by default each element is scaled by `alpha` if
    either its real part is strictly negative or if its real part is zero and its
    imaginary part is negative. This behaviour can be changed by specifying a different
    `complex_mode`.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.
    complex_mode
        optional specifier for how to handle complex data types. See
        `ivy.func_wrapper.handle_complex_input` for more detail.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.leaky_relu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.leaky_relu(x, out=y)
    >>> print(y)
    ivy.array([ 1.5 ,  0.7 , -0.48])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> ivy.leaky_relu(x, out=x)
    >>> print(x)
    ivy.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([0., -0.24000001]),
        b: ivy.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).leaky_relu(x, alpha=alpha, out=out)


leaky_relu.jax_like = _leaky_relu_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def log_softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the log_softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension log_softmax would be performed on. The default is ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output array with log_softmax applied element-wise to input.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, -0.98])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-0.703, -0.683])

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-2.41, -1.41, -0.408])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1.5, 0.5, 1.0])
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    ivy.array([-0.68, -1.68, -1.18])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.5, 0.5, 1.0]))
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    {
        a: ivy.array([-0.68, -1.68, -1.18])
    }

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0]), b=ivy.array([0.4, -0.2]))
    >>> y = ivy.log_softmax(x)
    >>> print(y)
    {
        a: ivy.array([-1.31, -0.313]),
        b: ivy.array([-0.437, -1.04])
    }
    """
    return current_backend(x).log_softmax(x, axis=axis, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def relu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the rectified linear unit function element-wise.

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
        an array containing the rectified linear unit activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.relu(x)
    >>> print(y)
    ivy.array([0., 0., 1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.relu(x, out = y)
    >>> print(y)
    ivy.array([1.5, 0.7, 0.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.relu(x, out=x)
    >>> print(x)
    {
        a: ivy.array([1., 0.]),
        b: ivy.array([0.40000001, 0.])
    }
    """
    return current_backend(x).relu(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.
        sigmoid activation of x is defined as 1/(1+exp(-x)).

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.zeros(3)
    >>> ivy.sigmoid(x,out=y)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.]),
                          b=ivy.Container(c=ivy.array([1.]),
                                          d=ivy.array([2.])))
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    {
        a: ivy.array([0.]),
        b: {
            c: ivy.array([1.]),
            d: ivy.array([2.])
        }
    }

    >>> x = ivy.Container(a=ivy.array([0.]),
                          b=ivy.Container(c=ivy.array([1.]),
                                          d=ivy.array([2.]))))
    >>> y = ivy.Container(a=ivy.array([0.]),
                          b=ivy.Container(c=ivy.array([0.]),
                                          d=ivy.array([0.]))))
    >>> ivy.sigmoid(x,out=y)
    >>> print(y)
    {
        a: ivy.array([0.]),
        b: {
            c: ivy.array([1.]),
            d: ivy.array([2.])
        }
    }
    """
    return current_backend(x).sigmoid(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def softmax(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with softmax applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.0, 0, 1.0])
    >>> y = ivy.softmax(x)
    >>> print(y)
    ivy.array([0.422, 0.155, 0.422])

    >>> x = ivy.array([[1.1, 2.2, 3.3],
    ...                [4.4, 5.5, 6.6]])
    >>> y = ivy.softmax(x, axis = 1)
    >>> print(y)
    ivy.array([[0.0768, 0.231 , 0.693 ],
               [0.0768, 0.231 , 0.693 ]])
    """
    return current_backend(x).softmax(x, axis=axis, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def softplus(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply the softplus function element-wise.

    Parameters
    ----------
    x
        input array.
    beta
        The beta value for the softplus formation. Default: ``None``.
    threshold
        values above this revert to a linear function. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x)
    >>> print(y)
    ivy.array([0.535,0.42])

    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softplus(x, beta=0.5)
    >>> print(y)
    ivy.array([1.22, 1.09])

    >>> x = ivy.array([1., 2., 3.])
    >>> y = ivy.softplus(x, threshold=2)
    >>> print(y)
    ivy.array([1.31, 2.13, 3.  ])
    """
    return current_backend(x).softplus(x, beta=beta, threshold=threshold, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def mish(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the mish activation function element-wise.

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
        an array containing the mish activation of each element in
        ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.mish(x)
    >>> print(y)
    ivy.array([-0.30340147,  0.        ,  0.86509842])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.mish(x, out = y)
    >>> print(y)
    ivy.array([ 1.40337825,  0.56114835, -0.20788449])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.4, -0.2]))
    >>> x = ivy.mish(x)
    >>> print(x)
    {
        a: ivy.array([0.86509842, -0.30883577]),
        b: ivy.array([0.28903052, -0.10714479])
    }
    """
    return current_backend(x).mish(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def hardswish(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Apply the hardswish activation function element-wise.

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
        an array containing the hardswish activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0., 0., 4.])
    >>> y = ivy.hardswish(x)
    >>> y
    ivy.array([0., 0., 4.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-3., 4., 5.]), b=ivy.array([0., 5.]))
    >>> x = ivy.hardswish(x, out=x)
    >>> x
    {
        a: ivy.array([-0.,  4.,  5.]),
        b: ivy.array([0., 5.])
    }
    """
    return current_backend(x).hardswish(x, out=out)
