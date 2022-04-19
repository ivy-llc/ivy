"""
Collection of Ivy loss functions.
"""

# local
import ivy


# Extra #
# ------#

def cross_entropy(true, pred, axis=-1, epsilon=1e-7):
    """Computes cross entropy between predicted and true discrete distrubtions.

    Parameters
    ----------
    true : array
        True labels
    pred : array
        predicted labels.
    axis : int, optional
        The class dimension, default is -1.
    epsilon : float, optional
        small constant to add to log functions, default is 1e-7

    Returns
    -------
    type
        The cross entropy loss

    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    # noinspection PyUnresolvedReferences
    return -ivy.sum(log_pred * true, axis)


# noinspection PyUnresolvedReferences
def binary_cross_entropy(true, pred, epsilon=1e-7):
    """Computes the binary cross entropy loss.

    Parameters
    ----------
    true : array
        true labels
    pred : array
        Predicted labels
    epsilon : float, optional
        small constant to add to log functions, default is 1e-7

    Returns
    -------
    type
        The binary cross entropy loss array.

    """
    pred = ivy.clip(pred, epsilon, 1-epsilon)
    # noinspection PyTypeChecker
    return -(ivy.log(pred) * true + ivy.log(1 - pred) * (1 - true))


def sparse_cross_entropy(true, pred, axis=-1, epsilon=1e-7):
    """Computes sparse cross entropy between logits and labels.

    Parameters
    ----------
    true : array
        True labels as logits.
    pred : array
        predicted labels as logits.
    axis : int, optional
        The class dimension, default is -1.
    epsilon : float, optional
        small constant to add to log functions, default is 1e-7

    Returns
    -------
    type
        The sparse cross entropy loss

    """
    true = ivy.one_hot(true, pred.shape[axis])
    return cross_entropy(true, pred, axis, epsilon)
