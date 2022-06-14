"""Collection of Ivy normalization functions."""


# local
from typing import List, Union, Optional
import ivy
from ivy.func_wrapper import to_native_arrays_and_back


# Extra #
# ------#


@to_native_arrays_and_back
def layer_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    normalized_idxs: List[int],
    epsilon: float = ivy._MIN_BASE,
    scale: float = None,
    offset: float = None,
    new_std: float = 1.0,
    out: Optional[ivy.Array] = None,
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
        optional output array, for writing the result to.

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
           [0., 0., 0.]])

    >>> arr = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> norm = ivy.zeros((2, 3))
    >>> ivy.layer_norm(arr, [0], out=norm)
    >>> print(norm)
    ivy.array([[-1., -1., -1.],
           [ 1.,  1.,  1.]])

    >>> arr = ivy.array([[0.0976, -0.3452,  1.2740], \
        [0.1047,  0.5886,  1.2732], \
        [0.7696, -1.7024, -2.2518]])
    >>> norm = ivy.layer_norm(arr, [0, 1], epsilon=0.001, \
                 new_std=1.5, offset=0.5, scale=0.5)
    >>> print(norm)
    ivy.array([[ 0.58 ,  0.283,  1.37 ],
           [ 0.585,  0.909,  1.37 ],
           [ 1.03 , -0.628, -0.997]])

    With :code:`ivy.NativeArray` input:

    >>> arr = ivy.native_array([[3., 1.],[4., 12.]])
    >>> norm = ivy.layer_norm(arr, [0,1], new_std=1.25, offset=0.25, scale=0.3)
    >>> print(norm)
    ivy.array([[ 0.0707, -0.109 ],
           [ 0.16  ,  0.877 ]])
<<<<<<< HEAD
    
    With :code:`ivy.Container` input:

    >>> container = ivy.Container({'a': ivy.array([2., 3., 4.]), \
        'b': ivy.array([1.3, 2.11, 0.243])})
    >>> norm = ivy.layer_norm(container, [0], new_std=1.25, offset=0.2)
=======

    With a mix of :code:`ivy.Array` and :code:`ivy.Container` inputs:

    >>> arr = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> norm_idxs = ivy.Container({'a': [0], 'b': [1]})
    >>> norm = ivy.layer_norm(arr, norm_idxs, new_std=1.25, offset=0.2)
    >>> print(norm)
    {
        a: ivy.array([[-1.05, -1.05, -1.05],
                      [1.45, 1.45, 1.45]]),
        b: ivy.array([[-1.33, 0.2, 1.73],
                      [-1.33, 0.2, 1.73]])
    }

    With :code:`ivy.Container` input:

    >>> arr = ivy.Container({'a': ivy.array([7., 10., 12.]), \
                            'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
    >>> norm_idxs = ivy.Container({'a': [0], 'b': [1]})
    >>> new_std = ivy.Container({'a': 1.25, 'b': 1.5})
    >>> offset = ivy.Container({'a': 0.2, 'b': 0.3})
    >>> norm = ivy.layer_norm(arr, norm_idxs, new_std, offset)
    >>> print(norm)
    {
        a: ivy.array([-0.228, 0.0285, 0.199]),
        b: ivy.array([[-0.204, 0., 0.204],
                      [-0.204, 0., 0.204]])
    }


    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> arr = ivy.array([[0.0976, -0.3452,  1.2740], \
        [0.1047,  0.5886,  1.2732], \
        [0.7696, -1.7024, -2.2518]])
    >>> norm = arr.layer_norm([0, 1], epsilon=0.001, \
                 new_std=1.5, offset=0.5, scale=0.5))
    >>> print(norm)
    ivy.array([[ 0.58 ,  0.283,  1.37 ],
           [ 0.585,  0.909,  1.37 ],
           [ 1.03 , -0.628, -0.997]])

    Using :code:`ivy.Container` instance method:

    >>> container = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> norm = container.layer_norm([0], new_std=1.25, offset=0.2)
>>>>>>> master
    >>> print(norm)
    {
        a: ivy.array([-1.33, 0.2, 1.73]),
        b: ivy.array([0.335, 1.66, -1.39])
    }

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> arr = ivy.array([[0.0976, -0.3452,  1.2740], \
        [0.1047,  0.5886,  1.2732], \
        [0.7696, -1.7024, -2.2518]])
    >>> norm = arr.layer_norm([0, 1], epsilon=0.001, \
                 new_std=1.5, offset=0.5, scale=0.5))
    >>> print(norm)
    ivy.array([[ 0.58 ,  0.283,  1.37 ],
           [ 0.585,  0.909,  1.37 ],
           [ 1.03 , -0.628, -0.997]])

    Using :code:`ivy.Container` instance method:

    >>> container = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
    >>> norm = container.layer_norm([0], new_std=1.25, offset=0.2)
    >>> print(norm)
    {
        a: ivy.array([-1.33, 0.2, 1.73]),
        b: ivy.array([0.335, 1.66, -1.39])
    }

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

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
    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x
