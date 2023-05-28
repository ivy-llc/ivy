import ivy
from typing import Optional, Union
from ivy.func_wrapper import (
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
)
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.ivy.losses import _reduce_loss


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

