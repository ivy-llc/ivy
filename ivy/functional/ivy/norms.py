"""Collection of Ivy normalization functions."""


# local
from typing import List, Union
import ivy


# Extra #
# ------#

# noinspection PyUnresolvedReferences
def layer_norm(
    x: Union[ivy.Array, ivy.Container, ivy.NativeArray],
    normalized_idxs: List[int],
    epsilon: float = ivy._MIN_BASE,
    scale=None,
    offset=None,
    new_std: float = 1.0,
) -> Union[ivy.Array, ivy.Container]:
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
    
    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> arr = ivy.full([2, 3], 13, dtype=ivy.float32)
    >>> norm = ivy.layer_norm(arr, [0, 1], new_std=2.0)
    >>> print(norm)
    ivy.array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    >>> arr = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> norm = ivy.layer_norm(arr, [0, 1])
    >>> print(norm)
    ivy.array([[-1.4638476 , -0.8783086 , -0.29276952],
           [ 0.29276952,  0.8783086 ,  1.4638476 ]], dtype=float32)

    >>> arr = ivy.array([[ 0.0976, -0.3452,  1.2740], \
        [ 0.1047,  0.5886,  1.2732], \
        [ 0.7696, -1.7024, -2.2518]])
    >>> norm = ivy.layer_norm(arr, [0, 1, 2], epsilon=0.001, \
                 new_std=1.5, offset=0.5, scale=0.5)
    >>> print(norm)
    ivy.array([[ 0.57629204,  0.29217   ,  1.3311275 ],
           [ 0.58084774,  0.89134157,  1.3306142 ],
           [ 1.0074799 , -0.5786756 , -0.9311974 ]], dtype=float32)

    With :code:`ivy.NativeArray` input:

    >>> tensor = ivy.native_array([[3.,1.],[4.,12.]])
    >>> norm = ivy.layer_norm(tensor, [0,1], new_std=1.25, offset=0.25, scale=0.3)
    >>> print(norm)
    ivy.array([[ 0.07071576, -0.10856849],
           [ 0.16035788,  0.8774949 ]], dtype=float32)
    
    With :code:`ivy.Container` input:

    >>> container = ivy.Container({'a': ivy.array([2.,3.,4.]),
    'b': ivy.array([1.3, 2.11, 0.243])})
    >>> norm = ivy.layer_norm(container, [0,1], new_std=1.25, offset=0.2)
    >>> print(norm)

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
