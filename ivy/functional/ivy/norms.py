"""Collection of Ivy normalization functions."""


# local
from typing import Tuple, Union, Optional
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    inputs_to_ivy_arrays,
    integer_arrays_to_float,
    handle_array_like_without_promotion,
    handle_nestable,
)
from ivy.utils.exceptions import handle_exceptions


# Extra #
# ------#


@inputs_to_ivy_arrays
@integer_arrays_to_float
@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_array_function
def layer_norm(
    x: Union[ivy.Array, ivy.NativeArray],
    normalized_shape: Tuple[int],
    /,
    *,
    scale: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    offset: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    eps: float = 1e-05,
    new_std: float = 1.0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies Layer Normalization over a mini-batch of inputs

    Parameters
    ----------
    x
        Input array
    normalized_shape
        Tuple containing the last k dimensions to apply normalization to.
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
    """
    feature_size = ivy.prod(normalized_shape)
    x_view = x.view((-1, feature_size))
    mean = ivy.mean(x_view, axis=-1, keepdims=True)
    var = ivy.var(x_view, axis=-1, keepdims=True)
    x_view = ivy.divide(
        ivy.add(ivy.negative(mean), x_view), ivy.stable_pow(var, 0.5, min_base=eps)
    )

    if scale is not None:
        if offset is not None:
            return ivy.multiply(
                ivy.add(
                    ivy.multiply(
                        x_view,
                        scale.view(
                            -1,
                        ),
                    ),
                    offset.view(
                        -1,
                    ),
                ),
                new_std,
                out=out,
            )
        return ivy.multiply(
            ivy.multiply(
                x_view,
                scale.view(
                    -1,
                ),
            ),
            new_std,
            out=out,
        )

    return ivy.multiply(x_view, new_std, out=out)
