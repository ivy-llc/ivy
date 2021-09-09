"""
Collection of Ivy normalization functions.
"""

# local
import ivy


# noinspection PyUnresolvedReferences
def layer_norm(x, normalized_idxs, epsilon=None, gamma=None, beta=None):
    """
    Applies Layer Normalization over a mini-batch of inputs

    :param x: Input array
    :type x: array
    :param normalized_idxs: Indices to apply the normalization to.
    :type normalized_idxs: int or sequence of ints
    :param epsilon: small constant to add to the denominator, use global ivy._MIN_BASE by default.
    :type epsilon: float, optional
    :param gamma: Learnable gamma variables for post-multiplication, default is None.
    :type gamma: array, optional
    :param beta: Learnable beta variables for post-addition, default is None.
    :type beta: array, optional
    :return: The layer after applying layer normalization.
    """
    mean = ivy.reduce_mean(x, normalized_idxs, keepdims=True)
    var = ivy.reduce_var(x, normalized_idxs, keepdims=True)
    x = (x - mean) / ivy.stable_pow(var, 0.5, epsilon)
    if gamma is not None:
        x = x * gamma
    if beta is not None:
        x = x + beta
    return x
