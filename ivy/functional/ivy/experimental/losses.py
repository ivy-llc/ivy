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


# Helper function for nce_loss


def log_uniform_candidate_sampler(
    true_classes,
    num_true,
    num_sampled,
    unique=True,
):
    pass


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def nce_loss(
    weights: Union[ivy.Array, ivy.NativeArray],
    biases: Union[ivy.Array, ivy.NativeArray],
    inputs: Union[ivy.Array, ivy.NativeArray],
    labels: Union[ivy.Array, ivy.NativeArray],
    num_sampled: ivy.int64,
    num_classes: ivy.int64,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits: bool = False,
    subtract_log_q: bool = True,
) -> ivy.Array:
    """
    Compute and returns the noise contrastive estimation loss. by default this uses a
    log-uniform (Zipfian) distribution for sampling, so your label must be sorted in
    order of decreasing frequency to achieve good results.

    Parameters
    ----------
    weights
        A 'Tensor' of shape '[num_classes,dim]', or a list of Tensor
        objects whose concatenation along dimension 0 has shape [num_classes,dim].
        The (possibly-partitiones) class embeddings.

    biases
        A 'Tensor' of shape '[num_classes]'. The class biases.

    labels
        A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    inputs
        A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled
        An `int`.  The number of negative classes to randomly sample
        per batch. This single sample of negative classes is evaluated for each
        element in the batch.
    num_classes
        An `int`. The number of possible classes.
    num_true
        An `int`.  The number of target classes per training example.
    sampled_values
        a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `log_uniform_sampler` function.
    remove_accidental_hits
        A `bool`.  Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  If set to
        `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
        learning to generate log-odds instead of log probabilities .


    Returns
    -------
    A 'batch_size' 1-D  tensor pf per-example NCE losses.
    """
    flat_labels = ivy.reshape(labels, [-1])

    if sampled_values is None:
        sampled_values = log_uniform_candidate_sampler(
            true_classes=labels,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes,
        )

    sampled, true_expected_count, sampled_expected_count = sampled_values

    all_ids = ivy.concat([flat_labels, sampled], axis=0)

    all_w = ivy.take_along_axis(weights, all_ids, axis=0)
    all_b = ivy.take_along_axis(weights, all_ids, axis=0)

    true_per_batch = labels.shape[0]

    true_w = all_w[:true_per_batch, :]
    true_b = all_b[:true_per_batch]

    sampled_w = all_w[true_per_batch:, :]
    sampled_b = all_w[true_per_batch:]

    # obtain true logits

    tw_c = true_w.shape[1]
    true_w = ivy.reshape(true_w, [-1, num_true, tw_c])

    row_wise_dots = ivy.multiply(ivy.expand_dims(inputs, 1), true_w)
    dot_as_matrix = ivy.reshape(row_wise_dots, [-1, tw_c])

    true_logits = ivy.reshape(ivy.sum(dot_as_matrix, 1), [-1, num_true])
    true_b = ivy.reshape(true_b, [-1, num_true])

    true_logits += true_b

    sampled_logits = ivy.matmul(inputs, ivy.matrix_transpose(sampled_w))
    sampled_logits += sampled_b

    if subtract_log_q:
        true_logits -= ivy.log(true_expected_count)
        sampled_logits -= ivy.log(sampled_expected_count)

    out_logits = ivy.concat([true_logits, sampled_logits], 1)

    out_labels = ivy.concat(
        [ivy.ones_like(true_logits) / num_true, ivy.zeros_like(sampled_logits)], 1
    )

    loss = ivy.binary_cross_entropy()

    nce_loss = loss(ivy.sigmoid(out_logits), out_labels)

    return nce_loss
