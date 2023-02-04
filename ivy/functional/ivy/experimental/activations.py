# global
from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_nestable,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional["ivy.Array"] = None,
) -> ivy.Array:
    """
    Computes the logit of x, i.e. logit(x) = log(x / (1 - x)).

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


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def prelu(
    x: Union[ivy.NativeArray, ivy.Array],
    slope: Union[float, ivy.NativeArray, ivy.Array],
    /,
    *,
    out: Optional["ivy.Array"] = None,
) -> ivy.Array:
    """
    Prelu takes input data (Array) and slope array as input,
    and produces one output data (array) where the function
    f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
    to the data array elementwise. This operator supports unidirectional
    broadcasting (array slope should be unidirectional broadcastable to
    input tensor X); for more details please check Broadcasting in ONNX.

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
    except ValueError as e:
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
        raise e


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def thresholded_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the rectified linear unit function with custom threshold.

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
