"""
Collection of Ivy normalization functions.
"""

# local
import ivy


# noinspection PyUnresolvedReferences
def layer_norm(x, normalized_idxs, epsilon=None, scale=None, offset=None, new_std=None):
    """
    Applies Layer Normalization over a mini-batch of inputs

    :param x: Input array
    :type x: array
    :param normalized_idxs: Indices to apply the normalization to.
    :type normalized_idxs: int or sequence of ints
    :param epsilon: small constant to add to the denominator, use global ivy._MIN_BASE by default.
    :type epsilon: float, optional
    :param scale: Learnable gamma variables for post-multiplication, default is None.
    :type scale: array, optional
    :param offset: Learnable beta variables for post-addition, default is None.
    :type offset: array, optional
    :param new_std: The standard deviation of the new normalized values. Default is 1.
    :type new_std: float, optional
    :return: The layer after applying layer normalization.
    """
    mean = ivy.reduce_mean(x, normalized_idxs, keepdims=True)
    var = ivy.reduce_var(x, normalized_idxs, keepdims=True)
    x = ((-mean + x) / ivy.stable_pow(var, 0.5, epsilon))
    if new_std is not None:
        x = x * new_std
    if scale is not None:
        x = x * scale
    if offset is not None:
        x = x + offset
    return x
