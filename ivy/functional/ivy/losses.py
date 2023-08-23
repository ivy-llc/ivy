"""Collection of Ivy loss functions."""

# local
import ivy
from typing import Optional, Union
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
    loss = loss.sum(axis=axis)
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
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    from_logits: bool = False,
    epsilon: float = 0.0,
    reduction: str = "none",
    pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the binary cross entropy loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    from_logits
        Whether `pred` is expected to be a logits tensor. By
        default, we assume that `pred` encodes a probability distribution.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``0``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    pos_weight
        a weight for positive examples. Must be an array with length equal to the number
        of classes.
    axis
        Axis along which to compute crossentropy.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary cross entropy between the given distributions.


    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 0, 0])
    >>> y = ivy.array([0.2, 0.8, 0.3, 0.8])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223,0.223,0.357,1.61])

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = ivy.binary_cross_entropy(x, y, reduction='mean')
    >>> print(z)
    ivy.array(7.6666193)

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = ivy.array([1, 2, 3, 4])
    >>> z = ivy.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True)
    ivy.array([[2.67164493e+00, 4.05471958e-03, 7.32684899e-02, 5.30496836e+00]])

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = ivy.array([1, 2, 3, 4])
    >>> z = ivy.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True, reduction='sum', axis=1) # noqa: E501
    ivy.array([8.05393649])

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = ivy.binary_cross_entropy(x, y, reduction='none', epsilon=0.5)
    ivy.array([[11.49992943,  3.83330965,  3.83330965, 11.49992943]])

    >>> x = ivy.array([[0, 1, 0, 0]])
    >>> y = ivy.array([[0.6, 0.2, 0.7, 0.3]])
    >>> z = ivy.binary_cross_entropy(x, y, epsilon=1e-3)
    >>> print(z)
    ivy.array([[0.916,1.61,1.2,0.357]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 0, 1])
    >>> y = ivy.native_array([0.2, 0.7, 0.2, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223,0.357,0.223,0.511])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0, 0, 1, 1])
    >>> y = ivy.native_array([0.1, 0.2, 0.8, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.105,0.223,0.223,0.511])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 0, 0]),b=ivy.array([0, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {a:ivy.array([0.511,0.223,0.357]),b:ivy.array([1.61,0.223,1.61])}

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1 , 1, 0])
    >>> y = ivy.Container(a=ivy.array([0.7, 0.8, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {
       a: ivy.array([0.357, 0.223, 0.223])
    }

    Instance Method Examples
    ------------------------

    Using :class:`ivy.Array` instance method:

    >>> x = ivy.array([1, 0, 0, 0])
    >>> y = ivy.array([0.8, 0.2, 0.2, 0.2])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223, 0.223, 0.223, 0.223])
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    true = true.astype(pred.dtype)

    epsilon = ivy.asarray(epsilon, dtype=pred.dtype)

    true = true * (1.0 - epsilon) + 0.5 * epsilon

    if from_logits:
        if pos_weight is not None:
            num_classes = pred.shape[0] if len(pred.shape) == 1 else pred.shape[1]
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    "pos_weight must have the same size as the number of classes in"
                    " pred at non-singleton dimension 1"
                )
            epsilon_ = 1e-7
            pred = ivy.sigmoid(pred)
            pred = ivy.clip(pred, epsilon_, 1 - epsilon_)
            loss = -(
                true * -ivy.log(pred) * pos_weight + (1 - true) * -ivy.log(1 - pred)
            )
        else:
            zeros = ivy.zeros_like(pred, dtype=pred.dtype)
            cond = pred >= zeros
            relu_logits = ivy.where(cond, pred, zeros)
            neg_abs_logits = ivy.where(cond, -pred, pred)
            loss = (
                ivy.add(relu_logits - pred * true, ivy.log1p(ivy.exp(neg_abs_logits)))
                * -1
            )
    else:
        epsilon_ = 1e-7
        pred = ivy.clip(pred, epsilon_, 1 - epsilon_)
        loss = true * ivy.log(pred + epsilon_) + (1 - true) * ivy.log(
            1 - pred + epsilon_
        )

    return _reduce_loss(reduction, loss, axis, out)
