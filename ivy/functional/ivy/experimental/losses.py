# global
from typing import Union

# local
import ivy
from ivy.func_wrapper import (
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    handle_array_function,
)
from ivy.utils.exceptions import handle_exceptions


# log_poisson_loss
@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def log_poisson_loss(
    targets: Union[ivy.Array, ivy.NativeArray],
    log_input: Union[ivy.Array, ivy.NativeArray],
    compute_full_loss: bool = False,
    name=None,
) -> ivy.Array:
    try:
        assert targets.shape == log_input.shape
    except ValueError:
        raise ValueError(
            "`log_input` and `targets` must have the same shape, received "
            f"({log_input.shape} vs {targets.shape})."
        )

    result = ivy.exp(log_input) - log_input * targets
    if compute_full_loss:
        point_five = 0.5
        two_pi = 2 * ivy.pi

        stirling_approx = (
            (targets * ivy.log(targets))
            - targets
            + (point_five * ivy.log(two_pi * targets))
        )
        zeros = ivy.zeros_like(targets, dtype=targets.dtype)
        ones = ivy.ones_like(targets, dtype=targets.dtype)
        cond = ivy.logical_and(targets >= zeros, targets <= ones)
        result += ivy.where(cond, zeros, stirling_approx)
    return result

@inputs_to_ivy_arrays
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def mse_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the mean squared error (MSE) loss.

    Parameters
    ----------
    true : array-like
        input array containing true values.
    pred : array-like
        input array containing predicted values.
    reduction : {'none', 'mean', 'sum'}, optional
        type of reduction to apply to the output. Default is 'mean'.
    out : array-like, optional
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    array
        The MSE loss between the true and predicted values.

    Raises
    -------
    ValueError
        If `reduction` is not one of ['none', 'mean', 'sum'].

    Examples
    --------
    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> ivy.mse_loss(true, pred)
    ivy.array(0.1075)

    >>> true = ivy.array([[1, 2], [3, 4]])
    >>> pred = ivy.array([[0.8, 2.3], [3.2, 4.1]])
    >>> ivy.mse_loss(true, pred, reduction='sum')
    ivy.array(0.17)

    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> out = ivy.array([0, 0])
    >>> ivy.mse_loss(true, pred, reduction='none', out=out)
    ivy.array([0.01 , 0.01 ])
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    result = ivy.mean(ivy.square(true - pred), keepdims=True)
    return _reduce_loss(reduction, result, None, out)


@inputs_to_ivy_arrays
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def mae_loss(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    reduction: str = "mean",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute the mean absolute error (MAE) loss.

    Parameters
    ----------
    true : array-like
        input array containing true values.
    pred : array-like
        input array containing predicted values.
    reduction : {'none', 'mean', 'sum'}, optional
        type of reduction to apply to the output. Default is 'mean'.
    out : array-like, optional
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    array
        The MAE loss between the true and predicted values.

    Raises
    -------
    ValueError
        If `reduction` is not one of ['none', 'mean', 'sum'].

    Examples
    --------
    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> ivy.mae_loss(true, pred)
    ivy.array(0.15)

    >>> true = ivy.array([[1, 2], [3, 4]])
    >>> pred = ivy.array([[0.8, 2.3], [3.2, 4.1]])
    >>> ivy.mae_loss(true, pred, reduction='sum')
    ivy.array(0.8)

    >>> true = ivy.array([1, 2, 3, 4])
    >>> pred = ivy.array([0.9, 2.1, 2.8, 4.2])
    >>> out = ivy.array([0, 0])
    >>> ivy.mae_loss(true, pred, reduction='none', out=out)
    ivy.array([0.1, 0.3])
    """
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    result = ivy.mean(ivy.abs(true - pred), keepdims=True)
    return _reduce_loss(reduction, result, None, out)
