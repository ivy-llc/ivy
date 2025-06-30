"""Collection of Ivy loss functions."""

# local
import ivy
from typing import Optional, Union
from ivy.func_wrapper import (
    handle_array_function,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions


# Helpers #
# ------- #


def _reduce_loss(red, loss, axis, out):
    if red == "sum":
        return ivy.negative(ivy.sum(loss, axis=axis), out=out)
    elif red == "mean":
        return ivy.negative(ivy.mean(loss, axis=axis), out=out)
    else:
        return ivy.negative(loss, out=out)


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute cross-entropy between predicted and true discrete distributions.

    Parameters
    ----------
    true
        input array containing true labels. Can be class indices of shape (N,) or 
        one-hot encoded labels of shape (N, C) where N is batch size and C is number of classes.
    pred
        input array containing the logits of shape (N, C).
    weight
        a manual rescaling weight given to each class. If given, has to be a array of size C.
    axis
        the axis along which to compute the cross-entropy. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating
        the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed.
        Default: ``'mean'``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The cross-entropy loss between the given distributions

    Examples
    --------
    >>> x = ivy.array([0, 0, 1, 0])
    >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
    >>> print(ivy.cross_entropy(x, y))
    ivy.array(1.3862944)

    >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.cross_entropy(x, z))
    ivy.array(0.9732134)
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if ivy.is_int_dtype(true) and len(true.shape) == len(pred.shape) - 1:
        num_classes = pred.shape[axis]
        true_one_hot = ivy.one_hot(true, num_classes, axis=axis)
    elif ivy.is_int_dtype(true) and len(true.shape) == len(pred.shape):
        if ivy.max(true) == 1 and ivy.min(true) == 0:
            true_one_hot = ivy.astype(true, pred.dtype)
        else:
            num_classes = pred.shape[axis]
            true_one_hot = ivy.one_hot(true, num_classes, axis=axis)
    else:
        true_one_hot = ivy.astype(true, pred.dtype)

    if epsilon > 0:
        num_classes = pred.shape[axis]
        true_one_hot = (1 - epsilon) * true_one_hot + epsilon / num_classes

    log_pred = ivy.log_softmax(pred, axis=axis)

    loss = -ivy.sum(true_one_hot * log_pred, axis=axis)

    if weight is not None:
        weight = ivy.asarray(weight, dtype=pred.dtype)
        if ivy.is_int_dtype(true) and len(true.shape) == len(pred.shape) - 1:
            sample_weights = ivy.gather(weight, true, axis=0)
            loss = loss * sample_weights
        else:
            if len(weight.shape) == 1:
                weight_shape = [1] * len(pred.shape)
                weight_shape[axis] = weight.shape[0]
                weight = ivy.reshape(weight, weight_shape)
            weighted_log_pred = log_pred * weight
            loss = -ivy.sum(true_one_hot * weighted_log_pred, axis=axis)

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        if weight is not None and ivy.is_int_dtype(true) and len(true.shape) == len(pred.shape) - 1:
            sample_weights = ivy.gather(weight, true, axis=0)
            total_weight = ivy.sum(sample_weights)
            if out is not None:
                return ivy.inplace_update(out, ivy.sum(loss) / ivy.clip(total_weight, 1e-8))
            return ivy.sum(loss) / ivy.clip(total_weight, 1e-8)
        else:
            return ivy.mean(loss, out=out)
    else:
        if out is not None: return ivy.inplace_update(out, loss)
        return loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def binary_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    from_logits: bool = False,
    epsilon: float = 0.0,
    reduction: str = "mean",
    pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the binary cross entropy loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    weight
        a manual rescaling weight if provided it's repeated to match input array shape.
    from_logits
        whether `pred` is expected to be a logits tensor. By
        default, we assume that `pred` encodes a probability distribution.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``0``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    pos_weight
        a weight for positive examples. Must be an array with length equal to the number
        of classes.
    axis
        axis along which to compute crossentropy.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        the binary cross entropy between the given distributions.


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 0, 0])
    >>> y = ivy.array([0.2, 0.8, 0.3, 0.8])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array(0.60309976)

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = ivy.binary_cross_entropy(x, y, reduction='mean')
    >>> print(z)
    ivy.array(7.6666193)

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = ivy.array([1, 2, 3, 4])
    >>> z = ivy.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True)
    ivy.array(2.01348412)

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = ivy.array([1, 2, 3, 4])
    >>> z = ivy.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True, reduction='sum', axis=1)
    >>> print(z)
    ivy.array([8.05393649])

    >>> x = ivy.array([[0, 1, 1, 0]])
    >>> y = ivy.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = ivy.binary_cross_entropy(x, y, reduction='none', epsilon=0.5)
    >>> print(z)
    ivy.array([[11.49992943,  3.83330965,  3.83330965, 11.49992943]])

    >>> x = ivy.array([[0, 1, 0, 0]])
    >>> y = ivy.array([[0.6, 0.2, 0.7, 0.3]])
    >>> z = ivy.binary_cross_entropy(x, y, epsilon=1e-3)
    >>> print(z)
    ivy.array(1.02136981)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 0, 1])
    >>> y = ivy.native_array([0.2, 0.7, 0.2, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array(0.32844672)

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0, 0, 1, 1])
    >>> y = ivy.native_array([0.1, 0.2, 0.8, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array(0.26561815)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 0, 0]),b=ivy.array([0, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {
        a: ivy.array(0.36354783),
        b: ivy.array(1.14733934)
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1 , 1, 0])
    >>> y = ivy.Container(a=ivy.array([0.7, 0.8, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {
       a: ivy.array(0.26765382)
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Using :class:`ivy.Array` instance method:

    >>> x = ivy.array([1, 0, 0, 0])
    >>> y = ivy.array([0.8, 0.2, 0.2, 0.2])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array(0.22314337)
    """  # noqa: E501
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    true = true.astype(pred.dtype)

    epsilon = ivy.asarray(epsilon, dtype=pred.dtype)

    true = true * (1.0 - epsilon) + 0.5 * epsilon

    if from_logits:
        if pos_weight is not None:
            num_classes = pred.shape[0] if len(pred.shape) == 1 else pred.shape[1]
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    "pos_weight must have the same size as the number of classes in"
                    " pred at non-singleton dimension 1"
                )
            epsilon_ = 1e-7
            pred = ivy.sigmoid(pred)
            pred = ivy.clip(pred, epsilon_, 1 - epsilon_)
            loss = -(
                true * -ivy.log(pred) * pos_weight + (1 - true) * -ivy.log(1 - pred)
            )
        else:
            zeros = ivy.zeros_like(pred, dtype=pred.dtype)
            cond = pred >= zeros
            relu_logits = ivy.where(cond, pred, zeros)
            neg_abs_logits = ivy.where(cond, -pred, pred)
            loss = (
                ivy.add(relu_logits - pred * true, ivy.log1p(ivy.exp(neg_abs_logits)))
                * -1
            )
    else:
        epsilon_ = 1e-7
        pred = ivy.clip(pred, epsilon_, 1 - epsilon_)
        loss = true * ivy.log(pred + epsilon_) + (1 - true) * ivy.log(
            1 - pred + epsilon_
        )

    if weight is not None: loss *= weight
    return _reduce_loss(reduction, loss, axis, out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def sparse_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute sparse cross entropy between logits and labels.

    Parameters
    ----------
    true
     input array containing the true labels as class indices.
    pred
     input array containing the predicted labels as logits.
    axis
     the axis along which to compute the cross-entropy. If axis is ``-1``, the
     cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
     a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
     loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    reduction
     ``'none'``: No reduction will be applied to the output.
     ``'mean'``: The output will be averaged.
     ``'sum'``: The output will be summed.
     Default: ``'mean'``.
    out
     optional output array, for writing the result to. It must have a shape
     that the inputs broadcast to.

    Returns
    -------
    ret
        The sparse cross-entropy loss between the given distributions

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([2])
    >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array(0.9732134)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([4])
    >>> y = ivy.native_array([0.1, 0.2, 0.1, 0.1, 0.5])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array(1.32223)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([4]))
    >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.1, 0.1, 0.5]))
    >>> print(ivy.sparse_cross_entropy(x, y))
    {
        a: ivy.array(1.32223)
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0])
    >>> y = ivy.native_array([0.1, 0.2, 0.6, 0.1])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array(1.5589634)

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([0])
    >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.6, 0.1]))
    >>> print(ivy.sparse_cross_entropy(x, y))
    {
        a: ivy.array(1.5589634)
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    With :class:`ivy.Array` input:

    >>> x = ivy.array([2])
    >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(x.sparse_cross_entropy(y))
    ivy.array(0.9732134)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([2]))
    >>> y = ivy.Container(a=ivy.array([0.1, 0.1, 0.7, 0.1]))
    >>> print(x.sparse_cross_entropy(y))
    {
        a: ivy.array(0.9732134)
    }
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    true = ivy.one_hot(true, pred.shape[axis])
    return ivy.cross_entropy(
        true, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def ssim_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Calculate the Structural Similarity Index (SSIM) loss between two
    images.

    Parameters
    ----------
        true: A 4D image array of shape (batch_size, channels, height, width).
        pred: A 4D image array of shape (batch_size, channels, height, width).

    Returns
    -------
        ivy.Array: The SSIM loss measure similarity between the two images.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> import ivy
    >>> x = ivy.ones((5, 3, 28, 28))
    >>> y = ivy.zeros((5, 3, 28, 28))
    >>> ivy.ssim_loss(x, y)
    ivy.array(0.99989986)
    """
    # Constants for stability
    C1 = 0.01**2
    C2 = 0.03**2

    # Calculate the mean of the two images
    mu_x = ivy.avg_pool2d(pred, (3, 3), (1, 1), "SAME")
    mu_y = ivy.avg_pool2d(true, (3, 3), (1, 1), "SAME")

    # Calculate variance and covariance
    sigma_x2 = ivy.avg_pool2d(pred * pred, (3, 3), (1, 1), "SAME") - mu_x * mu_x
    sigma_y2 = ivy.avg_pool2d(true * true, (3, 3), (1, 1), "SAME") - mu_y * mu_y
    sigma_xy = ivy.avg_pool2d(pred * true, (3, 3), (1, 1), "SAME") - mu_x * mu_y

    # Calculate SSIM
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    # Convert SSIM to loss
    ssim_loss_value = 1 - ssim

    # Return mean SSIM loss
    ret = ivy.mean(ssim_loss_value)

    if ivy.exists(out):
        ret = ivy.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def wasserstein_loss_discriminator(
    p_real: Union[ivy.Array, ivy.NativeArray],
    p_fake: Union[ivy.Array, ivy.NativeArray],
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Wasserstein loss for the discriminator (critic).

    Parameters
    ----------
        p_real (`ivy.Array`): Predictions for real data.
        p_fake (`ivy.Array`): Predictions for fake data.

    Returns
    -------
        `ivy.Array`: Wasserstein loss for the discriminator.
    """
    r_loss = ivy.mean(p_real)
    f_loss = ivy.mean(p_fake)
    ret = f_loss - r_loss

    if ivy.exists(out):
        ret = ivy.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def wasserstein_loss_generator(
    pred_fake: Union[ivy.Array, ivy.NativeArray],
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Wasserstein loss for the generator.

    Parameters
    ----------
        pred_fake (ivy.Array): Predictions for fake data.

    Returns
    -------
        ivy.Array: Wasserstein loss for the generator.
    """
    ret = -1 * ivy.mean(pred_fake)

    if ivy.exists(out):
        ret = ivy.inplace_update(out, ret)
    return ret
