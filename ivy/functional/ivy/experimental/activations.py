# global
from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.utils.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    integer_arrays_to_float,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
)


@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional[ivy.Array] = None,
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


@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def hardshrink(
    x: Union[float, int, ivy.Array],
    /,
    *,
    lambd: Optional[float] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Computes the hardshrink of x, i.e.
        hardshrink(x) = 0 if |x|<lambd else x

    Parameters
    ----------
    x
        Input container.
    lambd
        The value where the function is zero for inputs that are absolute value
        less than it.
    out
        Optional output container.

    Returns
    -------
    ret
        Container with hardshrink of the leaves.

    Examples
    --------
    >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
    >>> y = ivy.hardshrink(x)
    >>> print(y)
    ivy.array([ 0.69, -0.85,  0.,  0.])

    >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
    >>> y = ivy.hardshrink(x, lambd=0.2)
    >>> print(y)
    ivy.array([ 0.69, -0.85,  0.4,  0.])

    """
    return current_backend(x).hardshrink(x, lambd=lambd, out=out)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def softshrink(
    x: Union[float, int, ivy.Array],
    /,
    *,
    lambd: Optional[float] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Computes the softshrink of x, i.e.
        softshrink(x) = x + lambd  if x<-lambd
                        x - lambd if x>lambd
                        0        o/w

    Parameters
    ----------
    x
        Input container.
    lambd
        The value where the function is zero for inputs that are absolute value
        less than it.
    out
        Optional output container.

    Returns
    -------
    ret
        Container with softshrink of the leaves.

    Examples
    --------
    >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
    >>> y = ivy.softshrink(x)
    >>> print(y)
    ivy.array([ 0.19, -0.35,  0.,  0.])

    >>> x = ivy.array([0.69, -0.85, 0.4, -.2])
    >>> y = ivy.softshrink(x, lambd=0.2)
    >>> print(y)
    ivy.array([ 0.49, -0.65,  0.4,  0.])

    """
    return current_backend(x).softshrink(x, lambd=lambd, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def thresholded_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the thresholded rectified linear unit function with custom threshold.

    Parameters
    ----------
    x
        Input array
    threshold
        Threshold value above which the activation is linear. Default: ``0``.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the thresholded rectified linear unit activation 
        of each element in ``x``. with custom ``threshold``.

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


@handle_array_function
@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def threshold(
    x: Union[ivy.Array, ivy.NativeArray],
    threshold: Union[int, float],
    value: Union[int, float],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the threshold function with custom threshold and value.

    Parameters
    ----------
    x
        Input array
    threshold
        Threshold value above which the activation is linear.
    value
        The value to replace with
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the threshold function of each element in
        ``x``. with custom ``threshold`` and ``value``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([-1., 0., 1.])
    >>> y = ivy.threshold(x, 0.5, 2.)
    >>> print(y)
    ivy.array([2.,  2. ,  1.])

    >>> x = ivy.array([1.5, 0.7, -2.4])
    >>> y = ivy.zeros(3)
    >>> ivy.threshold(x, 1, -1., out=y)
    >>> print(y)
    ivy.array([ 1.5,  -1., -1.])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, -1.2]), b=ivy.array([0.2, 0.6]))
    >>> x = ivy.threshold(x, 0.5, 2.)
    >>> print(x)
    {
        a: ivy.array([1., 2.]),
        b: ivy.array([2., 0.6])
    }
    """
    return current_backend(x).threshold(x, threshold, value, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def relu6(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Applies the rectified linear unit 6 function element-wise.

    Parameters
    ----------
    x
        Input array
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the rectified linear unit 6 activation of each element in
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

    With :class:`ivy.Container` input:

    >>> x = {
                a: ivy.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
                b: ivy.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
            }
    >>> x = ivy.relu6(x, out=x)
    >>> print(x)
    {
    a: ivy.array([0., 0., 0., 0., 1., 2., 3., 4., 5.]),
    b: ivy.array([1., 2., 3., 4., 5., 6., 6., 6., 6.])
    }
    """
    return current_backend(x).relu6(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def batch_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 1e-5,
) -> ivy.Array:
    """
    Applies batch normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.
    variance
        A variance array for the input's normalization.
    offset
        An offset array. If present, will be added to the normalized input.
    scale
        A scale array. If present, the scale is applied to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
    )


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def group_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    num_groups: int,
    /,
    *,
    weight: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    bias: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    eps: float = 1e-5,
) -> ivy.Array:
    """
    Applies group normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    num_groups
        Number of groups to separate the channels into
    weight
        A scale array. If present, the scale is applied to the normalized input.
    bias
        An offset array. If present, will be added to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    shape = ivy.shape(x)
    assert shape[1] % num_groups == 0
    groups = shape[1] // num_groups
    num_dims = ivy.get_num_dims(x)
    expand_dims = (
        [0, *range(2, num_dims)] if weight is not None and num_dims > 2 else [0]
    )
    ret = ivy.concat(
        [
            ivy.layer_norm(
                x[:, i * groups : (i + 1) * groups, ...],
                list(range(1, num_dims)),
                scale=ivy.expand_dims(
                    weight[i * groups : (i + 1) * groups], axis=expand_dims
                )
                if weight is not None
                else None,
                b=ivy.expand_dims(bias[i * groups : (i + 1) * groups], axis=expand_dims)
                if bias is not None
                else None,
                epsilon=eps,
            )
            for i in range(num_groups)
        ],
        axis=1,
    )

    return ret


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def logsigmoid(
    x: Union[ivy.NativeArray, ivy.Array], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies element-wise Log-sigmoid of x i.e. logsigmoid(x) = log(1 / (1 + exp(-x)).

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

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
    return ivy.current_backend(x).logsigmoid(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the sigmoid function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        An array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])
    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).sigmoid(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the hard sigmoid function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        An array containing the sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.sigmoid()
    >>> print(y)
    ivy.array([0.269, 0.731, 0.881])
    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.sigmoid(x)
    >>> print(y)
    ivy.array([[0.214, 0.978, 0.891], [0.846,0.985,0.001]] )
    """
    return current_backend(x).hard_sigmoid(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def selu(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the scaled exponential linear units function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        An array containing the scaled exponential linear units activation
        of each element in ``x``.

    Examples
    --------

    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.selu(x)
    >>> print(y)
    ivy.array([-1.11,  1.05,  2.1])

    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.selu()
    >>> print(y)
    ivy.array([-1.11,  1.05,  2.1])

    >>> x = ivy.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = ivy.selu(x)
    >>> print(y)
    ivy.array([[-1.28,  3.99,  2.2], [ 1.79,  4.41, -1.76])

    """
    return current_backend(x).selu(x, out=out)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_tanh(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    min_value: float = -1.0,
    max_value: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Applies the hard tanh function element-wise.

    Parameters
    ----------
    x
        Input array.
    min_val
        Minimum value of the linear region range.
    max_val
        Maximum value of the linear region range
    out
        Optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        An array containing the hard tanh activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1. ,  1. ,  0.1])
    >>> y = ivy.hard_tanh(x)
    >>> print(y)
    ivy.array([-1. ,  1. ,  0.1])
    >>> x = ivy.array([-1. ,  1. ,  0.1])
    >>> y = x.hard_tanh()
    >>> print(y)
    ivy.array([-1. ,  1. ,  0.1])
    """
    return current_backend(x).hard_tanh(
        x, min_value=min_value, max_value=max_value, out=out
    )


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def log_sigmoid(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """
    Applies the log sigmoid function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        An array containing the log sigmoid activation of each element in ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = ivy.log_sigmoid(x)
    >>> print(y)
    ivy.array([-1.31, -0.31, -0.13])
    >>> x = ivy.array([-1.0, 1.0, 2.0])
    >>> y = x.log_sigmoid()
    >>> print(y)
    ivy.array([-1.31, -0.31, -0.13])
    """
    return current_backend(x).log_sigmoid(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def softsign(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the softsign function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the softsign activation of each element in ``x``.

    Functional Examples
    -------------------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.softsign(x)
    >>> print(y)
    ivy.array([0.2806, -0.4595])
    """
    return current_backend(x).softsign(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def silu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the silu function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the silu activation of each element in ``x``.

    Examples
    -------------------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.silu(x)
    >>> print(y)
    ivy.array([0.535, 0.42])
    """
    return current_backend(x).silu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def hard_silu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the hard silu function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array containing the hard silu activation of each element in ``x``.

    Examples
    -------------------
    With :class:`ivy.Array` input:
    >>> x = ivy.array([-0.3461, -0.6491])
    >>> y = ivy.hard_silu(x)
    >>> print(y)
    ivy.array([ 0.22, -0.3])
    """
    return current_backend(x).hard_silu(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def elu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the exponential linear unit activation function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Scalar or array of alpha values
    out
        Optional output array, for writing the result to. It must have a shape that the
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
    ivy.array([ 0.39, -0.17])

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.elu(x, alpha=2.0)
    >>> print(y)
    ivy.array([ 0.39, -0.34])

    """
    return current_backend(x).elu(x, alpha=alpha, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def parametric_relu(
    x: Union[ivy.Array, ivy.NativeArray],
    weight: Union[float, ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the parametric rectified linear unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with parametric relu applied element-wise.

    """
    return current_backend(x).parametric_relu(x, weight, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def celu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies the continuously-differentiable exponential linear unit activation
    function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for celu.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky continuously-differentiable exponential linear unit
        activation applied element-wise.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.celu(x)
    >>> print(y)
    ivy.array([ 0.39, -0.57])

    >>> x = ivy.array([0.39, -0.85])
    >>> y = ivy.celu(x, alpha=2.0)
    >>> print(y)
    ivy.array([ 0.39, -0.69])

    """
    return current_backend(x).celu(x, alpha=alpha, out=out)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def glu(
    x: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    axis: int = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Applies batch normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).glu(x, axis=axis, out=out)
