# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    handle_nestable,
    inputs_to_ivy_arrays,
    handle_array_like_without_promotion,
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


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def l1_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute L1 loss (Mean Absolute Error - MAE) between predicted and true values.

    Parameters
    ----------
    true : Union[ivy.Array, ivy.NativeArray]
        Input array containing true values.
    pred : Union[ivy.Array, ivy.NativeArray]
        Input array containing predicted values.
    reduction : str, optional
        Reduction method for the output loss. Options:
        "none" (no reduction), "mean" (mean of losses),
        "sum" (sum of losses). Default: "mean".
    out : Optional[ivy.Array], optional
        Optional output array for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ivy.Array
        The L1 loss (MAE) between the given true and predicted values.

    Examples
    --------
    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([0.5, 2.5, 2.0])
    >>> print(ivy.l1_loss(x, y))
    ivy.array(0.5)

    >>> a = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = ivy.array([[0.5, 1.5], [2.5, 3.5]])
    >>> print(ivy.l1_loss(a, b))
    ivy.array(1.0)
    """
    loss = ivy.abs(pred - true)

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, out=out)
    else:
        return loss
