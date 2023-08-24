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
    assumption that the target has a Poisson distribution. Caveat: By default, this is
    not the exact loss, but the loss minus a constant term [log(z!)]. That has no effect
    for optimization, but does not play well with relative loss comparisons. To compute
    an approximation of the log factorial term, specify ``compute_full_loss=True`` to
    enable Stirling's Approximation.

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
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: Optional[str] = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute L1 loss (Mean Absolute Error - MAE) between targeticted and input values.

    Parameters
    ----------
    input : Union[ivy.Array, ivy.NativeArray]
        Input array containing input values.
    target : Union[ivy.Array, ivy.NativeArray]
        Input array containing targeted values.
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
        The L1 loss (MAE) between the given input and targeticted values.


    Examples
    --------
    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([0.5, 2.5, 2.0])
    >>> print(ivy.l1_loss(x, y))
    ivy.array(0.6)
    >>> a = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = ivy.array([[0.5, 1.5], [2.5, 3.5]])
    >>> print(ivy.l1_loss(a, b))
    ivy.array(0.5)
    """
    loss = ivy.abs(target - input)

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def smooth_l1_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the smooth L1 loss between two input tensors.

    Parameters
    ----------
    input : array_like
        First input tensor.
    target : array_like
        Second input tensor.
    beta : float, optional
        The smooth parameter. Default is 1.0.
    reduction : str, optional
        Specifies the type of reduction to apply to the output.
        Should be one of 'none', 'sum', or 'mean'. Default is 'mean'.
    out : array, optional
        Optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret : array
        The smooth_l1_loss between the two input tensors.

    Examples
    --------
    >>> input = ivy.array([1.0, 2.0, 3.0])
    >>> target = ivy.array([2.5, 1.8, 3.2])
    >>> ivy.smooth_l1_loss(x, y, beta=1.0)
    ivy.array(0.3467)
    >>> input = ivy.array([1.0, 2.0, 3.0])
    >>> target = ivy.array([6.0, 2.0, 3.0])
    >>> ivy.smooth_l1_loss(x, y, beta=1.0)
    ivy.array(1.5)
    >>> input = ivy.array([2.0, 3.0, 5.0, 7.0])
    >>> target = ivy.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=1.5, reduction='sum')
    ivy.array(0.5)
    >>> input = ivy.array([0.8, 1.2, 2.5, 3.7])
    >>> target = ivy.array([0.9, 1.0, 2.3, 3.6])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=0.5, reduction='none')
    ivy.array([0.0133, 0.0250, 0.0056, 0.0025])
    >>> input = ivy.array([2.0, 3.0, 5.0, 7.0])
    >>> target = ivy.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=0.2, reduction='mean')
    ivy.array(0.025)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1.5, 2.2, 3.7])
    >>> y = ivy.native_array([2.1, 1.9, 3.5])
    >>> print(ivy.smooth_l1_loss(x, y, beta=0.5))
    ivy.array(0.0675)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]))
    >>> y = ivy.Container(a=ivy.array([2.5, 1.8, 3.2]))
    >>> print(ivy.smooth_l1_loss(x, y, beta=1.0))
    {
        a: ivy.array(0.3467)
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.native_array([6.0, 2.0, 3.0])
    >>> print(ivy.smooth_l1_loss(x, y, beta=0.5))
    ivy.array(1.5)

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.Container(a=ivy.array([6.0, 2.0, 3.0]))
    >>> print(ivy.smooth_l1_loss(x, y, beta=1.0))
    {
        a: ivy.array(1.5)
    }

    Instance Method Examples
    ------------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([2.5, 1.8, 3.2])
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    ivy.array(0.3467)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]))
    >>> y = ivy.Container(a=ivy.array([2.5, 1.8, 3.2]))
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    {
        a: ivy.array(0.3467)
    }
    """
    if beta < 1e-5:
        # if beta == 0,  will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = ivy.abs(input - target)
    else:
        n = ivy.abs(input - target)
        cond = n < beta
        loss = ivy.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        return ivy.mean(loss, out=out)
    elif reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "none":
        return ivy.inplace_update(out, loss) if out is not None else loss
