# local
import ivy
from typing import Optional, Union
from ivy.func_wrapper import (
    handle_nestable,
    handle_array_like_without_promotion,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_array_function,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.ivy.losses import _reduce_loss
from ivy.utils.backend import current_backend

@inputs_to_ivy_arrays
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def binary_cross_entropy_with_logits(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    epsilon: float = 1e-7,
    pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    reduction: str = "none",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the binary cross entropy with logits loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing predicted labels as logits.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    pos_weight
        a weight for positive examples. Must be an array with length equal to the number
        of classes.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary cross entropy with logits loss between the given distributions.


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 0, 1])
    >>> y = ivy.array([1.2, 3.8, 5.3, 2.8])
    >>> z = ivy.binary_cross_entropy_with_logits(x, y)
    >>> print(z)
    ivy.array([1.463, 0.022, 5.305, 0.059])

    >>> x = ivy.array([[0, 1, 0, 0]])
    >>> y = ivy.array([[6.6, 4.2, 1.7, 7.3]])
    >>> z = ivy.binary_cross_entropy_with_logits(x, y, epsilon=1e-3)
    >>> print(z)
    ivy.array([[6.601, 0.015, 1.868, 6.908]])

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = ivy.array([1.2])
    >>> z = ivy.binary_cross_entropy_with_logits(x, y, pos_weight=pos_weight)
    >>> print(z)
    ivy.array([[2.672, 0.002, 0.029, 5.305]])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0, 0, 0, 1])
    >>> y = ivy.native_array([3.1, 3.2, 1.8, 4.6])
    >>> z = ivy.binary_cross_entropy_with_logits(x, y)
    >>> print(z)
    ivy.array([3.144, 3.24, 1.953, 0.01])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 1, 0]), b=ivy.array([0, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([3.6, 1.2, 5.3]), b=ivy.array([1.8, 2.2, 1.2]))
    >>> z = ivy.binary_cross_entropy_with_logits(x, y)
    >>> print(z)
    {
        a: ivy.array([0.027, 0.263, 5.305]),
        b: ivy.array([1.953, 2.305, 0.263])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1, 0, 1])
    >>> y = ivy.Container(a=ivy.array([3.7, 3.8, 1.2]))
    >>> z = ivy.binary_cross_entropy_with_logits(x, y)
    >>> print(z)
    {
        a: ivy.array([0.024, 3.822, 0.263])
    }

    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    pred = ivy.sigmoid(pred)
    if pos_weight is not None:
        pred = ivy.clip(pred, epsilon, 1 - epsilon)
        result = -(true * -ivy.log(pred) * pos_weight + (1 - true) * -ivy.log(1 - pred))
        result = _reduce_loss(reduction, result, None, out)
    else:
        result = ivy.binary_cross_entropy(
            true,
            pred,
            epsilon=epsilon,
            reduction=reduction,
            out=out,
        )

    return result


@handle_array_function
@handle_nestable
@handle_exceptions
@handle_array_like_without_promotion
def ctc_loss(
        log_probs: Union[ivy.Array, ivy.NativeArray],
        targets: Union[ivy.Array, ivy.NativeArray],
        input_lengths: Union[ivy.Array, ivy.NativeArray],
        target_lengths: Union[ivy.Array, ivy.NativeArray],
        blank: int = 0,
        zero_infinity: bool = True,
        reduction: str = "mean",
        out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Computes the Connectionist Temporal Classification (CTC) loss between the log_probs and the targets
    Parameters
    ----------
    targets : ivy.Array, ivy.NativeArray
        Tensor of shape [batch_size, seq_len, num_labels] containing the log-probabilities of the predictions.
    log_probs : ivy.Array, ivy.NativeArray
        Tensor of shape [batch_size, max_target_len] containing the target labels.Each element must be a value
        in [0, num_labels-1]. Blank is assumed to be 0.
    target_lengths: ivy.Array, ivy.NativeArray
        Tensor of shape [batch_size] containing the lengths of each target sequence.
    input_lengths: ivy.Array, ivy.NativeArray
        Tensor of shape [batch_size] containing the lengths of each input sequence.
    blank : int
        Integer representing the blank label index. Default is 0.
    zero_infinity : bool
        Whether to zero infite losses and the associated gradients. Default is False.
    reduction : str
        The method to use to reduce the loss. It can be "mean", "sum", or "none". Default is "mean".
    out : ivy.Array, Optional
        Output array to write the result to.

    Returns
    -------
    ivy.Array
        Tensor of shape [batch_size] containing the CTC Loss for each input sequence.

    Raises
    -------
    ValueError
        If the shapes of pred, true, pred_lengths, and true_lengths are not compatible.
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    return current_backend(log_probs, targets, input_lengths, target_lengths,blank,zero_infinity,reduction,out).ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        zero_infinity=zero_infinity,
        reduction=reduction,
        out=out)

