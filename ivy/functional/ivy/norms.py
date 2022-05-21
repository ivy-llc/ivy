"""
Collection of Ivy normalization functions.
"""

# local
from ctypes import Union
from typing import Tuple, List
import ivy


# Extra #
# ------#

# noinspection PyUnresolvedReferences
def layer_norm(x: Union[ivy.Array, ivy.Container, ivy.NativeArray], normalized_idxs: List[int], epsilon: float=ivy._MIN_BASE, scale=None, offset=None, new_std: float=1.0)\
    -> Union[ivy.Array, ivy.Container]:
    """Applies Layer Normalization over a mini-batch of inputs

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
    
    Example
    -------

    >>> arr = ivy.full([2, 3], 13, dtype=ivy.float32)
    >>> arr = ivy.layer_norm(arr, [0, 1], new_std=2.0)
    >>> print(arr)
    [[-0.5  0.5  1.5]]

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> arr = ivy.layer_norm(arr, [0, 1])
    >>> print(arr)
    

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
