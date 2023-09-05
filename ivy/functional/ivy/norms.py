"""Collection of Ivy normalization functions."""


# local
from typing import List, Union, Optional
import ivy
from ivy.func_wrapper import (
    handle_array_like_without_promotion,
    handle_nestable,
    handle_array_function,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def layer_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    normalized_idxs: List[int],
    /,
    *,
    scale: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    offset: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    eps: float = 1e-05,
    new_std: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    x
        Input array
    normalized_idxs
        Indices to apply the normalization to.
    scale
        Learnable gamma variables for elementwise post-multiplication,
        default is ``None``.
    offset
        Learnable beta variables for elementwise post-addition, default is ``None``.
    eps
        small constant to add to the denominator. Default is ``1e-05``
    new_std
        The standard deviation of the new normalized values. Default is ``1``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        The layer after applying layer normalization.

    Examples
    --------
    With :class:`ivy.Array` input:
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
    >>> x = ivy.array([[0.0976, -0.3452,  1.2740],
    ...                [0.1047,  0.5886,  1.2732],
    ...                [0.7696, -1.7024, -2.2518]])
    >>> y = ivy.layer_norm(x, [0, 1], eps=0.001,
    ...                       new_std=1.5, scale=0.5, offset=[0.5, 0.02, 0.1])
    >>> print(y)
    ivy.array([[ 0.826, -0.178, 0.981 ],
               [ 0.831,  0.421, 0.981 ],
               [ 1.26 , -1.05 , -1.28 ]])
    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:
    >>> x = ivy.array([[1., 2., 3.], [4., 5., 6.]])
    >>> normalized_idxs = ivy.Container({'a': [0], 'b': [1]})
    >>> y = ivy.layer_norm(x, normalized_idxs, new_std=1.25, offset=0.2)
    >>> print(y)
    {
        a: ivy.array([[-1.25, -1.25, -1.25],
                      [1.25, 1.25, 1.25]]),
        b: ivy.array([[-1.53, 0., 1.53],
                      [-1.53, 0., 1.53]])
    }
    With one :class:`ivy.Container` input:
    >>> x = ivy.Container({'a': ivy.array([7., 10., 12.]),
    ...                    'b': ivy.array([[1., 2., 3.], [4., 5., 6.]])})
    >>> normalized_idxs = [0]
    >>> y = ivy.layer_norm(x, normalized_idxs, eps=1.25, scale=0.3)
    >>> print(y)
    {
        a: ivy.array([-0.34198591, 0.04274819, 0.29923761]),
        b: ivy.array([[-0.24053511, -0.24053511, -0.24053511],
                      [0.24053511, 0.24053511, 0.24053511]])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([7.0, 10.0, 12.0]),
    ...                   b=ivy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> normalized_idxs = ivy.Container(a=[0], b=[1])
    >>> new_std = ivy.Container(a=1.25, b=1.5)
    >>> bias = ivy.Container(a=[0.2, 0.5, 0.7], b=0.3)
    >>> y = ivy.layer_norm(x, normalized_idxs, new_std=new_std, offset=0.2)
    >>> print(y)
    {
        a: ivy.array([-1.62, 0.203, 1.42]),
        b: ivy.array([[-1.84, 0., 1.84],
                      [-1.84, 0., 1.84]])
    }
    # Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.
    """
    mean = ivy.mean(x, axis=normalized_idxs, keepdims=True)
    var = ivy.var(x, axis=normalized_idxs, keepdims=True)
    x = ivy.divide(
        ivy.add(ivy.negative(mean), x), ivy.stable_pow(var, 0.5, min_base=eps)
    )

    if scale is not None:
        if offset is not None:
            return ivy.multiply(
                ivy.add(ivy.multiply(x, scale), offset), new_std, out=out
            )
        return ivy.multiply(ivy.multiply(x, scale), new_std, out=out)

    return ivy.multiply(x, new_std, out=out)


layer_norm.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}
