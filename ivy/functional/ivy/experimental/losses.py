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


def _get_reduction_func(reduction):
    if reduction == "none":
        ret = lambda x: x
    elif reduction == "mean":
        ret = ivy.mean
    elif reduction == "sum":
        ret = ivy.sum
    else:
        raise ivy.utils.exceptions.IvyException(
            "{} is not a valid value for reduction".format(reduction)
        )
    return ret


def _legacy_get_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


def _get_reduction(reduction, size_average=None, reduce=None):
    if size_average is not None or reduce is not None:
        return _get_reduction_func(_legacy_get_string(size_average, reduce))
    else:
        return _get_reduction_func(reduction)


def _get_reduction_method(reduction, to_reduce):
    if reduction == "none":
        ret = to_reduce
    elif reduction == "mean":
        ret = ivy.mean(to_reduce)
    elif reduction == "sum":
        ret = ivy.sum(to_reduce)
    else:
        raise ivy.utils.exceptions.IvyException(
            f"{reduction} is not a valid value for reduction"
        )
    return ret


def _get_reduction_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


def _apply_reduction(reduction, size_average, reduce, to_reduce):
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
        return _get_reduction_method(reduction, to_reduce)
    else:
        return _get_reduction_method(reduction, to_reduce)


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


# hinge_embedding_loss
@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def hinge_embedding_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    margin: float = 1.0,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the Hinge Embedding Loss between the input and target.

    Parameters
    ----------
    input
        Model's raw output, typically logits or predicted scores for each sample.
    target
        Target variable representing true class labels.
        Should be a binary tensor where elements are either
        1 (positive class) or -1 (negative class).
    margin
        The margin value used in the hinge loss.
        The loss will be computed based on the difference between
        the predicted score and the margin for positive samples.
        Default: 1.0
    size_average
        Whether to average the loss across the samples or not.
        Default: None (delegates to the reduction parameter)
    reduce
        If True, the loss is summed over the samples. If False, it is returned as-is.
        Default: None (delegates to the reduction parameter)
    reduction
        'none': No reduction will be applied to the output.
        'mean': The output will be averaged.
        'sum': The output will be summed.
        Default: 'mean'
    out
      Optional output array, for writing the result to.
      It must have a
      shape that the inputs broadcast to.

    Returns
    -------
    ret
        The Hinge Embedding Loss between the input and target.

    Examples
    --------
    >>> x = ivy.array([0.5, -0.2, 1.2])
    >>> y = ivy.array([1, -1, 1])
    >>> print(ivy.hinge_embedding_loss(x, y))
    ivy.array(0.1)
    """
    margin = ivy.array(margin)

    loss = ivy.where(
        ivy.logical_or(target == -1, target == 1),
        ivy.where(target == 1, input, ivy.maximum(0, margin - input)),
        ivy.maximum(margin, input),
    )

    reduction = _get_reduction(reduction, size_average, reduce)
    ret = reduction(loss)

    return ivy.astype(ret, input.dtype)
