from typing import Union, Optional, Tuple

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_partial_mixed_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_array_function,
    handle_device_shifting,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def l1_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalize the input array along the given axis to have L1 norm equal to
    1.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis or axes along which to normalize. If ``None``,
         the whole array is normalized.
    out
        Optional output array, for writing the result to.
         It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        The normalized array.

    Examples
    --------
    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> y = ivy.l1_normalize(x, axis=1)
    >>> print(y)
    ivy.array([[0.33333334, 1.33333337],
           [1.28571439, 2.28571439]])
    """
    return current_backend(x).l1_normalize(x, axis=axis, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def l2_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalize the input array along the given axis to have L2 norm equal to
    1.

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
    >>> y = ivy.l2_normalize(x, axis=1)
    >>> print(y)
    ivy.array([[0.44721359, 0.89442718],
           [0.60000002, 0.80000001]])
    """
    return current_backend(x).l2_normalize(x, axis=axis, out=out)


@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def batch_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[Tuple[ivy.Array, ivy.Array, ivy.Array]] = None,
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
    """
    Apply batch normalization to the input array and returns the normalized input,
    running mean and running variance arrays as output. If ``training == False``,
    the mean and variance arrays passed as input are used for normalization
    and the same arrays are returned as running mean and running variance
    respectively. However, when ``training ==True``, this function computes the
    batch mean and batch variance which is then used for normalization.In this case,
    the function returns the running mean and running variance calculated
    using the following formula:

    running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    running_var = (1 - momentum) * running_var + momentum * frac{n}{n-1} * batch_var

    Parameters
    ----------
    x
        Input array of default shape (N, *S, C), where N is the batch dimension,
        *S corresponds to any number of spatial dimensions and
         C corresponds to the channel dimension.
    mean
        Mean array used for input's normalization. It can be of any shape
        braodcastable to (N,*S,C).
    variance
        Variance array used for input's normalization. It can be of any shape
        braodcastable to (N,*S,C).
    offset
        An offset array. If present, will be added to the normalized input.
        It can be of any shape broadcastable to (N,*S,C).
    scale
        A scale array. If present, the scale is applied to the normalized input.
        It can be of any shape broadcastable to (N,*S,C).
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.
    momentum
         the value used for the running_mean and running_var computation.
          Default value is 0.1.
    data_format
        The ordering of the dimensions in the input, one of "NSC" or "NCS",
        where N is the batch dimension, S represents any number of spatial
        dimensions and C is the channel dimension. Default is "NSC".
    out
        optional output arrays, for writing the result to.

    Returns
    -------
    ret
         Tuple of arrays containing
          the normalized input, running_mean, and running_variance.
    """
    xdims = len(x.shape)

    if data_format == "NCS":
        x = ivy.permute_dims(x, axes=(0, *range(2, xdims), 1))

    runningmean = mean
    runningvariance = variance

    if training:
        numel = int(ivy.prod(x.shape))
        n = numel if xdims == 1 else numel / x.shape[-1]
        dims = (0, *range(1, xdims - 1))
        mean = ivy.mean(x, axis=dims)
        variance = ivy.var(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (
            n - 1
        )
    inv = 1.0 / ivy.sqrt(variance + eps)
    if scale is not None:
        inv = inv * scale
    xnormalized = x * inv.astype(x.dtype, copy=False) + ivy.astype(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype
    )

    if data_format == "NCS":
        xnormalized = ivy.permute_dims(
            xnormalized, axes=(0, xdims - 1, *range(1, xdims - 1))
        )

    if ivy.exists(out):
        xnormalized = ivy.inplace_update(out[0], xnormalized)
        runningmean = ivy.inplace_update(out[1], runningmean)
        runningvariance = ivy.inplace_update(out[2], runningvariance)

    return xnormalized, runningmean, runningvariance


batch_norm.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def instance_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 0e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[Tuple[ivy.Array, ivy.Array, ivy.Array]] = None,
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
    """
    Apply instance normalization to the input array and returns the normalized input,
    running mean and running variance arrays as output. If ``training == False``,
    the mean and variance arrays passed as input are used for normalization
    and the same arrays are returned as running mean and running variance
    respectively. However, when ``training ==True``, this function computes the
    mean and variance across the spatial dimensions which is then used for
    normalization.In this case, the function returns the running mean and
    running variance calculated using the following formula:

    running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    running_var = (1 - momentum) * running_var + momentum * frac{n}{n-1} * batch_var

    Parameters
    ----------
    x
        Input array of default shape (N, *S, C), where N is the batch dimension,
        *S corresponds to any number of spatial dimensions and
         C corresponds to the channel dimension.
    mean
        Mean array of size C used for input's normalization.
    variance
        Variance array of size C used for input's normalization.
    offset
        An offset array of size C. If present, will be added
        to the normalized input.
    scale
        A scale array of size C. If present, the scale is
        applied to the normalized input.
    training
        If true, calculate and use the mean and variance of `x`.
        Otherwise, use the provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.
    momentum
         the value used for the running_mean and running_var computation.
          Default value is 0.1.
    data_format
        The ordering of the dimensions in the input, one of "NSC" or "NCS",
        where N is the batch dimension, S represents any number of spatial
        dimensions and C is the channel dimension. Default is "NSC".
    out
        optional output arrays, for writing the result to.

    Returns
    -------
    ret
         Tuple of arrays containing
          the normalized input, running_mean, and running_variance.
    """
    xdims = len(x.shape)
    if data_format == "NCS":
        x = ivy.permute_dims(x, axes=(*range(2, xdims), 0, 1))
    elif data_format == "NSC":
        x = ivy.permute_dims(x, axes=(*range(1, xdims - 1), 0, xdims - 1))
    else:
        raise ValueError(f"Invalid data_format: {data_format}.")

    N = x.shape[-2]
    C = x.shape[-1]
    S = x.shape[0:-2]
    x = x.reshape((1, *S, N * C))
    mean = ivy.tile(mean, N)
    variance = ivy.tile(variance, N)
    scale = ivy.tile(scale, N)
    offset = ivy.tile(offset, N)
    xnormalized, runningmean, runningvariance = batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    xnormalized = xnormalized.reshape((*S, N, C))

    if data_format == "NCS":
        xnormalized = ivy.permute_dims(
            xnormalized, axes=(xdims - 2, xdims - 1, *range(0, xdims - 2))
        )
    else:
        xnormalized = ivy.permute_dims(
            xnormalized, axes=(xdims - 2, *range(0, xdims - 2), xdims - 1)
        )

    runningmean = runningmean.reshape((N, C)).mean(axis=0)
    runningvariance = runningvariance.reshape((N, C)).mean(axis=0)

    if ivy.exists(out):
        xnormalized = ivy.inplace_update(out[0], xnormalized)
        runningmean = ivy.inplace_update(out[1], runningmean)
        runningvariance = ivy.inplace_update(out[2], runningvariance)

    return (xnormalized, runningmean, runningvariance)


instance_norm.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def group_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    num_groups: int = 1,
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    eps: Optional[float] = 1e-5,
    data_format: Optional[str] = "NSC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply group normalization to the input array and returns the normalized input.

    Parameters
    ----------
    x
        Input array of default shape (N, *S, C), where N is the batch dimension,
        *S corresponds to any number of spatial dimensions and
         C corresponds to the channel dimension.
    num_groups
        number of groups to separate the channels into
    offset
        An offset array of size C. If present, will be added
        to the normalized input.
    scale
        A scale array of size C. If present, the scale is
        applied to the normalized input.
    eps
        A small float number to avoid dividing by 0.
    data_format
        The ordering of the dimensions in the input, one of "NSC" or "NCS",
        where N is the batch dimension, S represents any number of spatial
        dimensions and C is the channel dimension. Default is "NSC".
    out
        optional output arrays, for writing the result to.

    Returns
    -------
    ret
        The normalized array.
    """
    xdims = ivy.get_num_dims(x)
    if data_format == "NSC":
        x = ivy.permute_dims(x, axes=(0, xdims - 1, *range(1, xdims - 1)))
    N = x.shape[0]
    C = x.shape[1]
    S = ivy.to_scalar(ivy.prod(x.shape[2:])) if xdims > 2 else 1
    assert C % num_groups == 0
    x_ = ivy.reshape(x, [N, num_groups, C // num_groups, S])
    mean = ivy.mean(x_, axis=(2, 3), keepdims=True)
    var = ivy.var(x_, axis=(2, 3), keepdims=True)
    x_normalized = (x_ - mean) / ivy.sqrt(var + eps)
    x_normalized = ivy.reshape(x_normalized, x.shape)

    if ivy.exists(scale):
        scale = ivy.expand_dims(scale, axis=[0, *(range(2, xdims))])
        x_normalized = x_normalized * scale

    if ivy.exists(offset):
        offset = ivy.expand_dims(offset, axis=[0, *(range(2, xdims))])
        x_normalized = x_normalized + offset

    if data_format == "NSC":
        x_normalized = ivy.permute_dims(x_normalized, axes=(0, *range(2, xdims), 1))

    if ivy.exists(out):
        x_normalized = ivy.inplace_update(out, x_normalized)
    return x_normalized


group_norm.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device_shifting
def lp_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalize the input array along the given axis to have Lp norm equal to
    1.

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
    >>> y = ivy.lp_normalize(x, p=1, axis=1)
    >>> print(y)
    ivy.array([[0.33333334, 0.66666669],
           [0.42857143, 0.5714286 ]])
    """
    return current_backend(x).lp_normalize(x, p=p, axis=axis, out=out)
