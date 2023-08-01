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
def gumbel_softmax(
    logits: Union[ivy.Array, ivy.NativeArray],
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> ivy.Array:
    """
    Sample from the Gumbel-Softmax distribution and optionally discretizes.

    Parameters
    ----------
    logits : array_like
        Unnormalized log probabilities of shape `[..., num_features]`.
    tau : float, optional
        Non-negative scalar temperature. Default: 1.0.
    hard : bool, optional
        If True, the returned samples will be discretized as one-hot vectors,
        but will be differentiated as if it is the soft sample in autograd.
        Default: False.
    eps : float, optional
        Small value to prevent numerical instability when
        taking logarithms. Default: 1e-10.
    dim : int, optional
        A dimension along which softmax will be computed. Default: -1.

    Returns
    -------
    array_like
        Sampled tensor of the same shape as `logits` from the
        Gumbel-Softmax distribution.
        If `hard=True`, the returned samples will be one-hot, otherwise they will be
        probability distributions that sum to 1 across `dim`.

    Examples
    --------
    >>> logits = ivy.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    >>> # Sample soft categorical using reparametrization trick:
    >>> soft_samples = ivy.gumbel_softmax(logits, tau=0.5, hard=False)
    >>> print(soft_samples)
    ivy.array([[0.217, 0.301, 0.482],
               [0.253, 0.346, 0.401]])
    >>> # Sample hard categorical using "Straight-through" trick:
    >>> hard_samples = ivy.gumbel_softmax(logits, tau=0.5, hard=True)
    >>> print(hard_samples)
    ivy.array([[0., 0., 1.],
               [0., 0., 1.]])
    """
    gumbels = -ivy.empty_like(logits).exponential().log()
    gumbels = (logits + gumbels) / tau
    y_soft = ivy.softmax(gumbels, axis=dim)

    if hard:
        indices = ivy.argmax(y_soft, axis=dim, keepdims=True)
        y_hard = ivy.zeros_like(logits)
        updates = ivy.ones_like(indices)
        y_hard = ivy.scatter_nd(indices, updates, reduction="replace", out=y_hard)

        ret = y_hard - ivy.stop_gradient(y_soft, preserve_type=True) + y_soft
    else:
        ret = y_soft

    return ret
