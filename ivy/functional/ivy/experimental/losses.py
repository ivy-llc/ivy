# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    handle_nestable,
    inputs_to_ivy_arrays,
    handle_array_like_without_promotion,
    handle_array_function,
    to_native_arrays_and_back,
)
from ivy.utils.exceptions import handle_exceptions


# log_poisson_loss
@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def log_poisson_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    compute_full_loss: bool = False,
    axis: int = -1,
    reduction: str = "none",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the log-likelihood loss between the prediction and the target
    under the assumption that the target has a Poisson distribution. Caveat: By
    default, this is not the exact loss, but the loss minus a constant term
    [log(z!)]. That has no effect for optimization, but does not play well with
    relative loss comparisons. To compute an approximation of the log factorial
    term, specify ``compute_full_loss=True`` to enable Stirling's
    Approximation.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    compute_full_loss
        whether to compute the full loss. If false, a constant term is dropped
        in favor of more efficient optimization. Default: ``False``.
    axis
        the axis along which to compute the log-likelihood loss. If axis is ``-1``,
        the log-likelihood loss will be computed along the last dimension.
        Default: ``-1``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary log-likelihood loss between the given distributions.


    Examples
    --------
    >>> x = ivy.array([0, 0, 1, 0])
    >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
    >>> print(ivy.log_poisson_loss(x, y))
    ivy.array([1.28402555, 1.28402555, 1.03402555, 1.28402555])

    >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.log_poisson_loss(x, z, reduction='mean'))
    ivy.array(1.1573164)
    """
    try:
        assert true.shape == pred.shape
    except ValueError as e:
        raise ValueError(
            "`pred` and `true` must have the same shape, received "
            f"({pred.shape} vs {true.shape})."
        ) from e

    loss = ivy.exp(pred) - pred * true
    if compute_full_loss:
        stirling_approx = (
            (true * ivy.log(true)) - true + (0.5 * ivy.log(2 * ivy.pi * true))
        )
        cond = ivy.logical_and(true >= 0.0, true <= 1.0)
        loss += ivy.where(cond, ivy.zeros_like(loss), stirling_approx)
    if reduction == "sum":
        return ivy.sum(loss, axis=axis, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, axis=axis, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def l1_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute L1 loss (Mean Absolute Error - MAE) between targeticted and input values.

    Parameters
    ----------
    input : Union[ivy.Array, ivy.NativeArray]
        Input array containing input values.
    target : Union[ivy.Array, ivy.NativeArray]
        Input array containing targeted values.
    reduction : str, optional
        Reduction method for the output loss. Options:
        "none" (no reduction), "mean" (mean of losses),
        "sum" (sum of losses). Default: "mean".
    out : Optional[ivy.Array], optional
        Optional output array for writing the result to.
        It must have a shape that the inputs broadcast to.


    Returns
    -------
    ivy.Array
        The L1 loss (MAE) between the given input and targeticted values.


    Examples
    --------
    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([0.5, 2.5, 2.0])
    >>> ivy.l1_loss(x, y)
    ivy.array(0.666)
    >>> a = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = ivy.array([[0.5, 1.5], [2.5, 3.5]])
    >>> ivy.l1_loss(a, b)
    ivy.array(0.5)
    """
    loss = ivy.abs(target - input)

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def huber_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    delta: float = 1.0,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Huber loss (smooth L1 loss) between true and predicted
    values.

    Parameters
    ----------
    true: array_like
        The true (ground truth) values.
    pred : array_like
        The predicted values by the model.
    delta : float, optional
        The threshold parameter that determines the point where the loss transitions fro
        -m
        squared error to absolute error. Default is 1.0.
    reduction : str, optional
        The type of reduction to apply to the loss. Possible values are "mean" (default)
        and "sum".
    out : array_like, optional
        Optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret : array_like
        The Huber loss between the true and predicted values.

    Examples
    --------
    >>> true = ivy.array([2, 4, 7, 1])
    >>> pred = ivy.array([2.5, 3.5, 8, 0.8])
    >>> huber_loss(true, pred, delta=1.0)
    ivy.array([0.125, 0.125, 0.5  , 0.125])

    >>> huber_loss(true, pred, delta=2.0)
    ivy.array([0.125, 0.125, 0.5  , 0.2  ])

    >>> huber_loss(true, pred, delta=0.5)
    ivy.array([0.25 , 0.25 , 0.   , 0.125])
    """
    abs_diff = ivy.abs(true - pred)
    quadratic_loss = 0.5 * (abs_diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = ivy.where(abs_diff <= delta, quadratic_loss, linear_loss)

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def smooth_l1_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    beta: float = 1.0,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the smooth L1 loss between two input tensors.

    Parameters
    ----------
    input : array_like
        First input tensor.
    target : array_like
        Second input tensor.
    beta : float, optional
        The smooth parameter. Default is 1.0.
    reduction : str, optional
        Specifies the type of reduction to apply to the output.
        Should be one of 'none', 'sum', or 'mean'. Default is 'mean'.
    out : array, optional
        Optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret : array
        The smooth_l1_loss between the two input tensors.

    Examples
    --------
    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([2.5, 1.8, 3.2])
    >>> ivy.smooth_l1_loss(x, y, beta=1.0)
    ivy.array(0.3467)
    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([6.0, 2.0, 3.0])
    >>> ivy.smooth_l1_loss(x, y, beta=1.0)
    ivy.array(1.5)
    >>> input = ivy.array([2.0, 3.0, 5.0, 7.0])
    >>> target = ivy.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=1.5, reduction='sum')
    ivy.array(0.5)
    >>> input = ivy.array([0.8, 1.2, 2.5, 3.7])
    >>> target = ivy.array([0.9, 1.0, 2.3, 3.6])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=0.5, reduction='none')
    ivy.array([0.0133, 0.0250, 0.0056, 0.0025])
    >>> input = ivy.array([2.0, 3.0, 5.0, 7.0])
    >>> target = ivy.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = ivy.smooth_l1_loss(input, target, beta=0.2, reduction='mean')
    ivy.array(0.025)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1.5, 2.2, 3.7])
    >>> y = ivy.native_array([2.1, 1.9, 3.5])
    >>> print(ivy.smooth_l1_loss(x, y, beta=0.5))
    ivy.array(0.0675)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]))
    >>> y = ivy.Container(a=ivy.array([2.5, 1.8, 3.2]))
    >>> print(ivy.smooth_l1_loss(x, y, beta=1.0))
    {
        a: ivy.array(0.3467)
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.native_array([6.0, 2.0, 3.0])
    >>> print(ivy.smooth_l1_loss(x, y, beta=0.5))
    ivy.array(1.5)

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.Container(a=ivy.array([6.0, 2.0, 3.0]))
    >>> print(ivy.smooth_l1_loss(x, y, beta=1.0))
    {
        a: ivy.array(1.5)
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.0, 2.0, 3.0])
    >>> y = ivy.array([2.5, 1.8, 3.2])
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    ivy.array(0.3467)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]))
    >>> y = ivy.Container(a=ivy.array([2.5, 1.8, 3.2]))
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    {
        a: ivy.array(0.3467)
    }
    """
    if beta < 1e-5:
        # if beta == 0,  will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = ivy.abs(input - target)
    else:
        n = ivy.abs(input - target)
        cond = n < beta
        loss = ivy.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        return ivy.mean(loss, out=out)
    elif reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "none":
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def soft_margin_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the soft-margin hinge loss between predicted scores and true
    binary labels.

    Parameters
    ----------
    input : array_like
        True binary labels, of shape (batch_size,).
    target : array_like
        Predicted scores, of shape (batch_size,).
    reduction : {'mean', 'sum', 'none'}, optional
        Type of reduction to apply to the output. Default is 'mean'.
    out : array_like, optional
        Optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret : array
        The soft-margin hinge loss between the predicted scores
        and true binary labels.

    Examples
    --------
    >>> input = ivy.array([1, 0, 1, 0])
    >>> target = ivy.array([0.8, 0.2, -0.6, 1.5])
    >>> ivy.soft_margin_loss(input, target)
    ivy.array(0.6987)

    >>> input = ivy.array([1, 1, 0, 0])
    >>> target = ivy.array([0.8, 0.7, 0.2, 0.1])
    >>> ivy.soft_margin_loss(input, target, reduction='sum')
    ivy.array(2.1606)

    >>> input = ivy.array([1, 1, 0, 0])
    >>> target = ivy.array([0.8, 0.7, 0.2, 0.1])
    >>> ivy.soft_margin_loss(input, target, reduction='none')
    ivy.array([0.3711, 0.4032, 0.6931, 0.6931])
    """
    loss = ivy.sum(ivy.log1p(ivy.exp(-input * target))) / input.size

    if reduction == "sum":
        return ivy.sum(loss, out=out)
    elif reduction == "mean":
        return ivy.mean(loss, out=out)
    else:
        return ivy.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
def kl_div(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: str = "mean",
    log_target=False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the Kullback-Leibler divergence loss between two input tensors
    (conventionally, probability distributions).

    Parameters
    ----------
    input : array_like
        Tensor of arbitrary shape in log-probabilities
    target : array_like
        Tensor of the same shape as input. See log_target for
        the target’s interpretation
    reduction : {'mean', 'sum', 'batchmean', 'none'}, optional
        Type of reduction to apply to the output. Default is 'mean'.
    log_target : bool
        A flag indicating whether target is passed in the log space.
        It is recommended to pass certain distributions (like softmax)
        in the log space to avoid numerical issues caused by explicit log.
        Default: False

    Returns
    -------
    ret : array
        The Kullback-Leibler divergence loss between the two input tensors.

    Examples
    --------
    >>> input = ivy.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = ivy.array([[0.6, 0.4], [0.3, 0.7]])
    >>> ivy.kl_div(input, target)
    ivy.array(-0.555969)

    >>> input = ivy.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = ivy.array([[0.6, 0.4], [0.3, 0.7]])
    >>> ivy.kl_div(input, target, reduction='sum')
    ivy.array(-2.223876)

    >>> input = ivy.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = ivy.array([[0.6, 0.4], [0.3, 0.7]])
    >>> ivy.kl_div(input, target, reduction='batchmean')
    ivy.array(-1.111938)

    >>> input = ivy.array([0.2, 0.8], [0.5, 0.5])
    >>> target = ivy.array([0.6, 0.4], [0.3, 0.7])
    >>> ivy.kl_div(input, target, reduction='none')
    ivy.array([[-0.42649534, -0.68651628],
                [-0.51119184, -0.59967244]])
    """
    if not log_target:  # default
        loss_pointwise = target * (ivy.log(target) - input)
    else:
        loss_pointwise = ivy.exp(target) * (target - input)

    if reduction == "mean":  # default
        loss = ivy.mean(loss_pointwise)
    elif reduction == "batchmean":  # mathematically correct
        loss = ivy.sum(loss_pointwise) / input.shape[0]
    elif reduction == "sum":
        loss = ivy.sum(loss_pointwise)
    else:  # reduction == "none"
        loss = loss_pointwise
    return ivy.inplace_update(out, loss) if out is not None else loss


kl_div.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_out_argument",
    ),
    "to_skip": ("inputs_to_ivy_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def poisson_nll_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> ivy.Array:
    r"""Compute the Poisson Negative Log Likelihood Loss.

    This function calculates the negative log likelihood loss
    between the `input` and `target`under the assumption that
    the target follows a Poisson distribution. By default, the loss
    is not the exact loss, but the loss minus a constant term [log(z!)].
    This omission does not affect optimization but can be significant for
    relative loss comparisons. The Stirling's Approximation is used to
    approximate the log factorial term when `full` is set to True.

    Parameters
    ----------
    input
        Expectation of the underlying Poisson distribution.
    target
        Random sample from the Poisson distribution described by the input.
    log_input
        If `True`, the loss is computed as
        :math:`exp(input) - target * input`. If `False`, the loss is computed as
        :math:`input - target * log(input + eps)`. Default is `True`.
    full
        Whether to compute the full loss, i.e., to add the Stirling approximation term
        :math:`target * log(target) - target + 0.5 * log(2 * pi * target)`.
        Default is `False`.
    eps
        Small value to prevent evaluation of `log(0)` when `log_input` is `False`.
        Default is 1e-8.
    reduction
        Specifies the reduction applied to the output.
        Options are 'none', 'mean', or 'sum'.
        'none': no reduction will be applied.
        'mean': the output will be averaged.
        'sum': the output will be summed.
        Default is 'mean'.

    Returns
    -------
    ret
        An array of the same shape as `input` representing
        the Poisson Negative Log Likelihood Loss.

    Raises
    ------
    ValueError
        If the `input` and `target` tensors do not have the same shape.

    Examples
    --------
    >>> input_tensor = ivy.array([1, 2, 3, 4], dtype=ivy.float64)
    >>> target_tensor = ivy.array([2, 2, 2, 2], dtype=ivy.float64)
    >>> loss = ivy.poisson_nll_loss(input_tensor, target_tensor, log_input=False)
    >>> print(loss)
    ivy.array(0.91097307)
    """
    return ivy.current_backend().poisson_nll_loss(
        input,
        target,
        log_input=log_input,
        full=full,
        eps=eps,
        reduction=reduction,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def hinge_embedding_loss(
    input: Union[ivy.Array, ivy.NativeArray],
    target: Union[ivy.Array, ivy.NativeArray],
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> ivy.Array:
    r"""Measures loss from input `x` and label `y` with values 1 or -1. It
    evaluates if two inputs are similar or not, often used for embedding or
    semi-supervised learning.

    Loss for the `n`-th sample:
        .. math::
            l_n = \begin{cases}
                x_n, & \text{if}\; y_n = 1,\\
                \max \{0, margin - x_n\}, & \text{if}\; y_n = -1,
            \end{cases}

    Total loss:
        .. math::
            \ell(x, y) = \begin{cases}
                \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
                \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
            \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top` .

    Parameters
    ----------
    input
        Input tensor with dtype float.
        The shape is [N, \*], where N is batch size and `\*` represents
        any number of additional dimensions.
    label
        Label tensor containing 1 or -1 with dtype float32 or float64.
        Its shape matches that of the input.
    margin
        Sets the hyperparameter margin. Determines the necessary input size
        for hinge_embedding_loss calculations when label is -1. Inputs smaller
        than the margin are minimized with hinge_embedding_loss.
        Default is 1.0.
    reduction
        Specifies how to aggregate the loss across the batch. Options are:
        - ``'none'``: Returns the unreduced loss.
        - ``'mean'``: Returns the mean loss.
        - ``'sum'``: Returns the summed loss.
        Default is ``'mean'``.

    Shape
    -----
        - Input: :math:`(*)` where :math:`*` means, any number of dimensions. \
        The sum operation operates over all the elements.
        - Target: :math:`(*)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``,
        then same shape as the input

    Returns
    -------
    ret
        Hinge embedding loss calculated from the input and label,
        shaped based on the reduction method.

    Examples
    --------
    >>> input_tensor = ivy.array([1, 2, 3, 4], dtype=ivy.float64)
    >>> target_tensor = ivy.array([1, 1, 1, 1], dtype=ivy.float64)
    >>> loss = ivy.hinge_embedding_loss(input_tensor, target_tensor, reduction="none")
    >>> loss
    ivy.array([1., 2., 3., 4.])

    >>> input_tensor = ivy.array([21, 22], dtype=ivy.float32)
    >>> target_tensor = ivy.array([-1, 1], dtype=ivy.float32)
    >>> loss = ivy.hinge_embedding_loss(input_tensor,target_tensor,
    ...                                 margin=2.0, reduction="sum")
    >>> loss
    ivy.array(22.)
    """
    return ivy.current_backend().hinge_embedding_loss(
        input,
        target,
        margin=margin,
        reduction=reduction,
    )
