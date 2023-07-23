# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    handle_nestable,
    inputs_to_ivy_arrays,
    handle_array_function,
)
from ivy.utils.exceptions import handle_exceptions


# log_poisson_loss
@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def log_poisson_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    compute_full_loss: bool = False,
    axis: int = -1,
    reduction: str = "none",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the log-likelihood loss between the prediction and the target under the
    assumption that the target has a Poisson distribution. Caveat: By default,
    this is not the exact loss, but the loss minus a constant term [log(z!)].
    That has no effect for optimization, but does not play well with relative loss
    comparisons. To compute an approximation of the log factorial term, specify
    ``compute_full_loss=True`` to enable Stirling's Approximation.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    compute_full_loss
        whether to compute the full loss. If false, a constant term is dropped
        in favor of more efficient optimization. Default: ``False``.
    axis
        the axis along which to compute the log-likelihood loss. If axis is ``-1``,
        the log-likelihood loss will be computed along the last dimension.
        Default: ``-1``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary log-likelihood loss between the given distributions.


    Examples
    --------
    >>> x = ivy.array([0, 0, 1, 0])
    >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
    >>> print(ivy.log_poisson_loss(x, z))
    ivy.array([1.28402555, 1.28402555, 1.03402555, 1.28402555])

    >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.log_poisson_loss(x, z, reduction='mean'))
    ivy.array(1.1573164)
    """
    try:
        assert true.shape == pred.shape
    except ValueError:
        raise ValueError(
            "`pred` and `true` must have the same shape, received "
            f"({pred.shape} vs {true.shape})."
        )

    loss = ivy.exp(pred) - pred * true
    if compute_full_loss:
        stirling_approx = (
            (true * ivy.log(true)) - true + (0.5 * ivy.log(2 * ivy.pi * true))
        )
        cond = ivy.logical_and(true >= 0.0, true <= 1.0)
        loss += ivy.where(cond, ivy.zeros_like(loss), stirling_approx)
    if reduction == "sum":
        return ivy.sum(loss, axis=axis, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, axis=axis, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@inputs_to_ivy_arrays
@handle_nestable
@handle_exceptions
def mse_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the mean squared error (MSE) loss.

    Parameters
    ----------
    true : array-like
        input array containing true values.
    pred : array-like
        input array containing predicted values.
    axis : int or None, optional
        the axis along which to compute the mean squared error. If `axis` is `None`,
        the mean squared error will be computed over all dimensions.
        Default is `None`.
    reduction : {'none', 'mean', 'sum'}, optional
        type of reduction to apply to the output. Default is 'mean'.
    out : array-like, optional
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    array
        The MSE loss between the true and predicted values.

    Raises
    -------
    ValueError
        If `reduction` is not one of ['none', 'mean', 'sum'].

    Examples
    --------
    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> ivy.mse_loss(true, pred)
    ivy.array(0.1075)

    >>> true = ivy.array([[1, 2], [3, 4]])
    >>> pred = ivy.array([[0.8, 2.3], [3.2, 4.1]])
    >>> ivy.mse_loss(true, pred, reduction='sum')
    ivy.array(0.17)

    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> out = ivy.array([0, 0])
    >>> ivy.mse_loss(true, pred, reduction='none', out=out)
    ivy.array([0.01 , 0.01 ])
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    result = ivy.mean(ivy.square(true - pred), axis=axis, keepdims=True)
    return _reduce_loss(reduction, result, None, out)


@inputs_to_ivy_arrays
@handle_nestable
@handle_exceptions
def mae_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the mean absolute error (MAE) loss.

    Parameters
    ----------
    true : array-like
        input array containing true values.
    pred : array-like
        input array containing predicted values.
    axis : int or None, optional
        the axis along which to compute the mean absolute error. If `axis` is `None`,
        the mean absolute error will be computed over all dimensions.
        Default is `None`.
    reduction : {'none', 'mean', 'sum'}, optional
        type of reduction to apply to the output. Default is 'mean'.
    out : array-like, optional
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    array
        The MAE loss between the true and predicted values.

    Raises
    -------
    ValueError
        If `reduction` is not one of ['none', 'mean', 'sum'].

    Examples
    --------
    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([1.1, 2.2, 2.9, 4.1])
    >>> ivy.mae_loss(true, pred)
    ivy.array(0.275)

    >>> true = ivy.array([[1, 2], [3, 4]])
    >>> pred = ivy.array([[0.9, 2.1], [3.1, 4.2]])
    >>> ivy.mae_loss(true, pred, reduction='sum')
    ivy.array(0.3)

    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([1.1, 2.2, 2.9, 4.1])
    >>> out = ivy.array([0, 0])
    >>> ivy.mae_loss(true, pred, reduction='none', out=out)
    ivy.array([0.1, 0.2])
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    result = ivy.mean(ivy.abs(true - pred), axis=axis, keepdims=True)
    return _reduce_loss(reduction, result, None, out)