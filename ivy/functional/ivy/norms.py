"""Collection of Ivy normalization functions."""


# local
from typing import List, Union, Optional
import ivy
from ivy.func_wrapper import handle_nestable


# Extra #
# ------#


@handle_nestable
def layer_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    normalized_idxs: List[int],
    epsilon: float = ivy._MIN_BASE,
    scale: float = 1.0,
    offset: float = 0.0,
    new_std: float = 1.0,
    *,
    out: Optional[ivy.Array] = None
) -> ivy.Array:
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        The layer after applying layer normalization.
    
    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = ivy.layer_norm(x, [0, 1], new_std=2.0)
    >>> print(y)
    ivy.array([[-2.68 , -0.894],
               [ 0.894,  2.68 ]])

    >>> x = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> y = ivy.zeros((2, 3))
    >>> ivy.layer_norm(x, [0], out=y)
    >>> print(y)
    ivy.array([[-1., -1., -1.],
               [ 1.,  1.,  1.]])

    >>> x = ivy.array([[0.0976, -0.3452,  1.2740], \
                       [0.1047,  0.5886,  1.2732], \
                       [0.7696, -1.7024, -2.2518]])
    >>> y = ivy.layer_norm(x, [0, 1], epsilon=0.001, \
                              new_std=1.5, offset=0.5, scale=0.5)
    >>> print(y)
    ivy.array([[ 0.576,  0.292,  1.33 ],
               [ 0.581,  0.891,  1.33 ],
               [ 1.01 , -0.579, -0.931]])

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> x = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
    >>> y = ivy.layer_norm(x, normalized_idxs, new_std=1.25, offset=0.2)
    >>> print(y)
    {
        a: ivy.array([[-1.05, -1.05, -1.05],
                      [1.45, 1.45, 1.45]]),
        b: ivy.array([[-1.33, 0.2, 1.73],
                      [-1.33, 0.2, 1.73]])
    }

    With one :code:`ivy.Container` input:

    >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                           'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
    >>> normalized_idxs = [0]
    >>> y = ivy.layer_norm(x, normalized_idxs, 1.25, 0.3)
    >>> print(y)
    {
        a: ivy.array([0.658, 1.04, 1.3]),
        b: ivy.array([[0.759, 0.759, 0.759], 
                      [1.24, 1.24, 1.24]])
    }

    With multiple :code:`ivy.Container` inputs:

    >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                           'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
    >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
    >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
    >>> offset = ivy.Container({'a': 0.2, 'b': 0.3})
    >>> y = ivy.layer_norm(x, normalized_idxs, new_std, offset)
    >>> print(y)
    {
        a: ivy.array([0.772, 1.03, 1.2]),
        b: ivy.array([[0.796, 1., 1.2], 
                      [0.796, 1., 1.2]])
    }

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    """
    mean = ivy.mean(x, normalized_idxs, keepdims=True)
    var = ivy.var(x, normalized_idxs, keepdims=True)
    x = ivy.divide(ivy.add(ivy.negative(mean), x), ivy.stable_pow(var, 0.5, epsilon))
    return ivy.add(ivy.multiply(ivy.multiply(x, new_std), scale), offset, out=out)
