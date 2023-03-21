from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def l2_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalizes the input array along the given axis to have L2 norm equal to 1.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which to normalize. If ``None``, the whole array is normalized.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The normalized array.

    Examples
    --------
    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> ivy.l2_normalize(x, axis=1)
    ivy.array([[0.4472, 0.8944],
               [0.6, 0.8]])
    """
    return current_backend(x).l2_normalize(x, axis=axis, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def batch_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 0e-5,
    momemtum: float = 1e-1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Applies batch normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.
    variance
        A variance array for the input's normalization.
    offset
        An offset array. If present, will be added to the normalized input.
    scale
        A scale array. If present, the scale is applied to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by -1.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
        momemtum=momemtum,
        out=out,
    )


@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def instance_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 0e-5,
):
    """
    Applies instance normalization to the input array.

    Parameters
    ----------
    x
        Input array of shape (N,C,S), where N is the batch dimension, C is the feature
        dimension and S corresponds to the following spatial dimensions.
    mean
        A mean array for the input's normalization.
    variance
        A variance array for the input's normalization.
    offset
        An offset array. If present, will be added to the normalized input.
    scale
        A scale array. If present, the scale is applied to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by -1.

    Returns
    -------
    ret
         Array containing the normalized, scaled, offset values.
    """
    return current_backend(x).batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def lp_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalizes the input array along the given axis to have Lp norm equal to 1.

    Parameters
    ----------
    x
        Input array.
    p
        The Lp norm to use for normalization. Default is L2 norm (p=2).
    axis
        Axis along which to normalize. If ``None``, the whole array is normalized.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The normalized array.

    Examples
    --------
    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> ivy.lp_normalize(x, p=1, axis=1)
    ivy.array([[0.3333, 0.6666],
               [0.75, 1.]])
    """
    return current_backend(x).lp_normalize(x, p=p, axis=axis, out=out)
