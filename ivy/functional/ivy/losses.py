"""
Collection of Ivy loss functions.
"""

# local
import ivy
from ivy import NativeArray, Array

from typing import Union, Optional


# Extra #
# ------#

def cross_entropy(true : Union[Array, NativeArray], pred : Union[Array, NativeArray], axis : Optional[int] = -1, epsilon : Optional[float] = 1e-7, \
    out: Optional[Union[Array, NativeArray]] = None) \
    -> Union[Array, NativeArray]:
    """
    Computes cross entropy between predicted and true discrete distrubtions.
    Note that ``true`` and ``pred`` must have the same shape.

    Parameters
    ----------
    true: array
        True labels
    pred: array
        predicted labels
    axis: int, optional
        The class dimension, default is ``-1``
    epsilon: float, optional
        small constant to add to log functions, default is ``1e-7``
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
        The cross entropy loss

    Examples
    -------
    >>> true = ivy.array([[1, 0, 0], [0, 1, 0]]) # shape (2, 3)
    >>> pred = ivy.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]) # shape (2, 3)
    >>> loss = ivy.cross_entropy(true, pred)
    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    # noinspection PyUnresolvedReferences
    ret = -ivy.sum(log_pred * true, axis)

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)

    return ret



# noinspection PyUnresolvedReferences
def binary_cross_entropy(true : Union[Array, NativeArray], pred : Union[Array, NativeArray], epsilon : Optional[float] = 1e-7, \
    out: Optional[Union[Array, NativeArray]] = None) \
    -> Union[Array, NativeArray]:
    """
    Computes the binary cross entropy loss. 
    Note that ``true`` and ``pred`` must have the same shape. 

    Parameters
    ----------
    true: array
        True labels
    pred: array
        predicted labels
    epsilon: float, optional
        small constant to add to log functions, default is ``1e-7``
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
        The binary cross entropy loss

    Examples
    -------
    >>> true = ivy.array([1, 0, 0, 1, 0])
    >>> pred = ivy.array([0.9, 0.1, 0.5, 0.8, 0.6])
    >>> loss = ivy.binary_cross_entropy(true, pred)
    """
    pred = ivy.clip(pred, epsilon, 1-epsilon)
    # noinspection PyTypeChecker
    ret = -(ivy.log(pred) * true + ivy.log(1 - pred) * (1 - true))

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)

    return ret


def sparse_cross_entropy(true : Union[Array, NativeArray], pred : Union[Array, NativeArray], axis : Optional[int] = -1, epsilon : Optional[float] = 1e-7, \
    out: Optional[Union[Array, NativeArray]] = None) \
    -> Union[Array, NativeArray]:
    """
    Computes sparse cross entropy between predicted and true discrete distrubtions.

    Parameters
    ----------
    true: array
        True labels
    pred: array
        predicted labels
    axis: int, optional
        The class dimension, default is ``-1``
    epsilon: float, optional
        small constant to add to log functions, default is ``1e-7``
    out:
        optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

    Returns
    -------
        The sparse cross entropy loss

    Examples
    -------
    >>> true = ivy.array([0, 2]) # shape (2,)
    >>> pred = ivy.array([[0.8, 0.1, 0.1], [0.2, 0.1, 0.7]]) # shape (2, 3)
    >>> loss = ivy.sparse_cross_entropy(true, pred)
    """
    true = ivy.one_hot(true, pred.shape[axis])
    ret = cross_entropy(true, pred, axis, epsilon)

    if ivy.exists(out):
        return ivy.inplace_update(out, ret)

    return ret
