"""Collection of Ivy loss functions."""

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions


# Helpers #
# ------- #


def _reduce_loss(
    loss, target, reduction="mean", class_weights=None, sample_weights=1.0, axis=-1
):
    if sample_weights is not None:
        loss = loss * sample_weights
    if reduction == "mean":
        if class_weights is not None:
            return -loss.sum() / (class_weights * target).sum()
        else:
            return -loss.mean()
    elif reduction == "sum":
        return -loss.sum()
    else:
        return -loss


def _broadcast_class_weights(class_weights, target, axis=-1):
    num_dims_to_extend = target.ndim - 1
    shape = [1 for x in range(num_dims_to_extend)] + [len(class_weights)]
    return ivy.reshape(class_weights, shape).swapaxes(-1, axis)


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def cross_entropy(
    inputs,
    target,
    from_logits=False,
    class_weights=None,
    sample_weights=None,
    reduction="mean",
    label_smoothing=0.0,
    axis=-1,
):
    log_probs = ivy.softmax(inputs, axis=axis).log() if from_logits else inputs.log()
    if inputs.ndim != target.ndim:
        target = target.astype("int32")
        _target = ivy.one_hot(target, target.max() + 1, axis=axis)
    else:
        _target = target

    if label_smoothing > 0.0:
        num_classes = _target.shape[axis]
        _target = (1.0 - label_smoothing) * _target + (label_smoothing / num_classes)

    loss = log_probs * _target
    if class_weights is not None:
        class_weights = _broadcast_class_weights(
            class_weights=class_weights, target=_target, axis=axis
        )
        loss = loss * class_weights

    loss = loss = loss.sum(axis=axis)

    return _reduce_loss(
        loss,
        _target,
        reduction=reduction,
        class_weights=class_weights,
        axis=axis,
        sample_weights=sample_weights,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def binary_cross_entropy(
    inputs,
    target,
    from_logits=False,
    sample_weights=None,
    reduction="mean",
    label_smoothing=0.0,
):
    probs = inputs.sigmoid() if from_logits else inputs
    log_probs_1 = probs.log()
    log_probs_0 = (1 - probs).log()

    if label_smoothing > 0.0:
        # this is slightly different from the label smoothing done in cross_entropy
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = target * log_probs_1 + (1 - target) * log_probs_0
    return _reduce_loss(
        loss,
        target,
        reduction=reduction,
        class_weights=None,
        sample_weights=sample_weights,
    )
