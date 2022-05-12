"""Collection of Ivy normalization functions."""

# local
import ivy


# Extra #
# ------#

# noinspection PyUnresolvedReferences
def layer_norm(x, normalized_idxs, epsilon=None, scale=None, offset=None, new_std=None):
    """Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    x
        Input array
    normalized_idxs
        Indices to apply the normalization to.
    epsilon
        small constant to add to the denominator, use global ivy._MIN_BASE by default.
    scale
        Learnable gamma variables for post-multiplication, default is None.
    offset
        Learnable beta variables for post-addition, default is None.
    new_std
        The standard deviation of the new normalized values. Default is 1.

    Returns
    -------
    ret
        The layer after applying layer normalization.

    """
    mean = ivy.mean(x, normalized_idxs, keepdims=True)
    var = ivy.var(x, normalized_idxs, keepdims=True)
    x = (-mean + x) / ivy.stable_pow(var, 0.5, epsilon)
    if new_std is not None:
        x = x * new_std
    if scale is not None:
        x = x * scale
    if offset is not None:
        x = x + offset
    return x
