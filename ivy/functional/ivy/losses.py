"""
Collection of Ivy loss functions.
"""

# local
import ivy


def cross_entropy(true, pred, axis=-1, epsilon=1e-7):
    """
    Computes cross entropy between predicted and true discrete distrubtions.

    :param true: True labels
    :type true: array
    :param pred: predicted labels.
    :type pred: array
    :param axis: The class dimension, default is -1.
    :type axis: int, optional
    :param epsilon: small constant to add to log functions, default is 1e-7
    :type epsilon: float, optional
    :return: The cross entropy loss
    """
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    # noinspection PyUnresolvedReferences
    return -ivy.reduce_sum(log_pred * true, axis)


# noinspection PyUnresolvedReferences
def binary_cross_entropy(true, pred, epsilon=1e-7):
    """
    Computes the binary cross entropy loss.

    :param true: true labels
    :type true: array
    :param pred: Predicted labels
    :type pred: array
    :param epsilon: small constant to add to log functions, default is 1e-7
    :type epsilon: float, optional
    :return: The binary cross entropy loss array.
    """
    pred = ivy.clip(pred, epsilon, 1-epsilon)
    # noinspection PyTypeChecker
    return -(ivy.log(pred) * true + ivy.log(1 - pred) * (1 - true))


def sparse_cross_entropy(true, pred, axis=-1, epsilon=1e-7):
    """
    Computes sparse cross entropy between logits and labels.

    :param true: True labels as logits.
    :type true: array
    :param pred: predicted labels as logits.
    :type pred: array
    :param axis: The class dimension, default is -1.
    :type axis: int, optional
    :param epsilon: small constant to add to log functions, default is 1e-7
    :type epsilon: float, optional
    :return: The sparse cross entropy loss
    """
    true = ivy.one_hot(true, pred.shape[axis])
    return cross_entropy(true, pred, axis, epsilon)
