"""Collection of Ivy loss functions."""

# local
import ivy
from typing import Optional, Union

# Extra #
# ------#


def cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = -1,
    epsilon: Optional[float] = 1e-7,
) -> ivy.Array:
    """Computes cross-entropy between predicted and true discrete distributions.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing the predicted labels.
    axis
        the axis along which to compute the cross-entropy. If axis is ``-1``,
        the cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating
        the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.

    Returns
    -------
    ret
        The cross-entropy loss between the given distributions

    Examples
    --------
    >>> x = ivy.array([0, 0, 1, 0])
    >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
    >>> print(ivy.cross_entropy(x, y))
    ivy.array(1.38629436)

    >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.cross_entropy(x, z))
    ivy.array(0.35667497)

    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    # noinspection PyUnresolvedReferences
    return -ivy.sum(log_pred * true, axis)


# noinspection PyUnresolvedReferences
def binary_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    pred: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    epsilon: Optional[float] = 1e-7,
) -> Union[ivy.Array, ivy.Container]:
    """Computes the binary cross entropy loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.

    Returns
    -------
    ret
        The binary cross entropy between the given distributions.

    """
    """
    Functional Examples
    --------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 0, 0])
    >>> y = ivy.array([0.2, 0.8, 0.3, 0.8])
    >>> print(ivy.binary_cross_entropy(x, y))
    ivy.array([0.2231, 0.2231, 0.3567, 1.6094])

    >>> z = ivy.array([0.6, 0.2, 0.7, 0.3])
    >>> print(ivy.binary_cross_entropy(x, z))
    ivy.array([0.9163, 1.6094, 1.2040, 0.3567])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 0, 1])
    >>> y = ivy.native_array([0.2, 0.7, 0.2, 0.6])
    >>> print(ivy.binary_cross_entropy(x, y))
    ivy.array([0.2231, 0.3567, 0.2231, 0.5108])

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0, 0, 1, 1])
    >>> y = ivy.native_array([0.1, 0.2, 0.8, 0.6])
    >>> print(ivy.binary_cross_entropy(x, y))
    ivy.array([0.1054, 0.2231, 0.2231, 0.5108])

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([1, 0, 0, 0])
    >>> y = ivy.array([0.8, 0.2, 0.2, 0.2])
    >>> print(ivy.binary_cross_entropy(x, y))
    ivy.array([0.2231, 0.2231, 0.2231, 0.2231]))

    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    # noinspection PyTypeChecker
    return -(ivy.log(pred) * true + ivy.log(1 - pred) * (1 - true))


def sparse_cross_entropy(true, pred, axis=-1, epsilon=1e-7):
    """Computes sparse cross entropy between logits and labels.

    Parameters
    ----------
    true
        True labels as logits.
    pred
        predicted labels as logits.
    axis
        The class dimension, default is -1.
    epsilon
        small constant to add to log functions, default is 1e-7

    Returns
    -------
    ret
        The sparse cross entropy loss

    """
    true = ivy.one_hot(true, pred.shape[axis])
    return cross_entropy(true, pred, axis, epsilon)
