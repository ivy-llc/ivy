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
    true
        True labels
    pred
        predicted labels.
    axis
        The class dimension, default is -1.
    epsilon
        small constant to add to log functions, default is 1e-7

    Returns
    -------
     ret
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
    pred = ivy.clip(pred, epsilon, 1-epsilon)
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
