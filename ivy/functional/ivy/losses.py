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
    *,
    out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Computes cross-entropy between predicted and true discrete distributions.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing the predicted labels.
    axis
        the axis along which to compute the cross-entropy. If axis is ``-1``, the
        cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The cross-entropy loss between the given distributions

    Examples
    --------
    >>> x = ivy.array([0, 0, 1, 0])
    >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
    >>> print(ivy.cross_entropy(x, y))
    ivy.array(1.3862944)

    >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.cross_entropy(x, z))
    ivy.array(0.35667497)

    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    # noinspection PyUnresolvedReferences
    return ivy.negative(ivy.sum(log_pred * true, axis), out=out)


# noinspection PyUnresolvedReferences
def binary_cross_entropy(true, pred, epsilon=1e-7):
    """Computes the binary cross entropy loss.

    Parameters
    ----------
    true
        true labels
    pred
        Predicted labels
    epsilon
        small constant to add to log functions, default is 1e-7

    Returns
    -------
    ret
        The binary cross entropy loss array.

    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    # noinspection PyTypeChecker
    return -(ivy.log(pred) * true + ivy.log(1 - pred) * (1 - true))


def sparse_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = -1,
    epsilon: Optional[float] = 1e-7,
) -> ivy.Array:
    """Computes sparse cross entropy between logits and labels.

    Parameters
    ----------
    true
        input array containing the true labels as logits.
    pred
        input array containing the predicted labels as logits.
    axis
        the axis along which to compute the cross-entropy. If axis is ``-1``, the
        cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.

    Returns
    -------
    ret
        The sparse cross-entropy loss between the given distributions

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([2])
    >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array([0.35667497 ])

    >>> x = ivy.array([3])
    >>> print(ivy.cross_entropy(x, y))
    ivy.array([2.3025851 ])

    >>> x = ivy.array([2,3])
    >>> print(ivy.cross_entropy(x, y))
    ivy.array([0.35667497, 2.3025851 ])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([4])
    >>> y = ivy.native_array([0.1, 0.2, 0.1, 0.1, 0.5])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array([0.693])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4]))
    >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.1, 0.1, 0.5]))
    >>> print(ivy.sparse_cross_entropy(x, y))
    {
        a: ivy.array([0.693])
    }

    With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0])
    >>> y = ivy.native_array([0.1, 0.2, 0.6, 0.1])
    >>> print(ivy.sparse_cross_entropy(x,y))
    ivy.array([2.3])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([0])
    >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.6, 0.1]))
    >>> print(ivy.sparse_cross_entropy(x,y))
    {
        a: ivy.array([2.3])
    }

    Instance Method Examples
    ------------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([2])
    >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(x.sparse_cross_entropy(y))
    ivy.array([0.357])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([2]))
    >>> y = ivy.Container(a=ivy.array([0.1, 0.1, 0.7, 0.1]))
    >>> print(x.sparse_cross_entropy(y))
    {
        a: ivy.array([0.357])
    }

    """
    true = ivy.one_hot(true, pred.shape[axis])
    return cross_entropy(true, pred, axis, epsilon)
