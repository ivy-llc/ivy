"""Collection of Ivy loss functions."""

# local
import ivy
from typing import Optional, Union
from ivy.func_wrapper import handle_nestable, handle_array_like
from ivy.exceptions import handle_exceptions

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


@handle_nestable
@handle_exceptions
@handle_array_like
def cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes cross-entropy between predicted and true discrete distributions.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing the predicted labels.
    axis
        the axis along which to compute the cross-entropy. If axis is ``-1``,
        the cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating
        the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
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
    ivy.array(0.35667497)

    """
    ivy.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    return _reduce_loss(reduction, log_pred * true, axis, out)


@handle_nestable
@handle_exceptions
@handle_array_like
def binary_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    epsilon: float = 1e-7,
    reduction: str = "none",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the binary cross entropy loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary cross entropy between the given distributions.


    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([0, 1, 0, 0])
    >>> y = ivy.array([0.2, 0.8, 0.3, 0.8])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223,0.223,0.357,1.61])

    >>> x = ivy.array([[0, 1, 0, 0]])
    >>> y = ivy.array([[0.6, 0.2, 0.7, 0.3]])
    >>> z = ivy.binary_cross_entropy(x, y, epsilon=1e-3)
    >>> print(z)
    ivy.array([[0.916,1.61,1.2,0.357]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0, 1, 0, 1])
    >>> y = ivy.native_array([0.2, 0.7, 0.2, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223,0.357,0.223,0.511])

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x = ivy.array([0, 0, 1, 1])
    >>> y = ivy.native_array([0.1, 0.2, 0.8, 0.6])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.105,0.223,0.223,0.511])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 0, 0]),b=ivy.array([0, 0, 1]))
    >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {a:ivy.array([0.511,0.223,0.357]),b:ivy.array([1.61,0.223,1.61])}

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([1 , 1, 0])
    >>> y = ivy.Container(a=ivy.array([0.7, 0.8, 0.2]))
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    {
       a: ivy.array([0.357, 0.223, 0.223])
    }

    Instance Method Examples
    ------------------------

    Using :class:`ivy.Array` instance method:

    >>> x = ivy.array([1, 0, 0, 0])
    >>> y = ivy.array([0.8, 0.2, 0.2, 0.2])
    >>> z = ivy.binary_cross_entropy(x, y)
    >>> print(z)
    ivy.array([0.223, 0.223, 0.223, 0.223])

    """
    ivy.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    return _reduce_loss(
        reduction,
        ivy.add(ivy.log(pred) * true, ivy.log(1 - pred) * (1 - true)),
        None,
        out,
    )


@handle_nestable
@handle_exceptions
@handle_array_like
def sparse_cross_entropy(
    true: Union[ivy.Array, ivy.NativeArray],
    pred: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes sparse cross entropy between logits and labels.

    Parameters
    ----------
    true
     input array containing the true labels as logits.
    pred
     input array containing the predicted labels as logits.
    axis
     the axis along which to compute the cross-entropy. If axis is ``-1``, the
     cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
     a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
     loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    out
     optional output array, for writing the result to. It must have a shape
     that the inputs broadcast to.

    Returns
    -------
    ret
        The sparse cross-entropy loss between the given distributions

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2])
    >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
    >>> print(ivy.sparse_cross_entropy(x, y))
    ivy.array([0.357])

     >>> x = ivy.array([3])
     >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
     >>> print(ivy.cross_entropy(x, y))
     ivy.array(21.793291)

     >>> x = ivy.array([2,3])
     >>> y = ivy.array([0.1, 0.1])
     >>> print(ivy.cross_entropy(x, y))
     ivy.array(11.512926)

     With :class:`ivy.NativeArray` input:

     >>> x = ivy.native_array([4])
     >>> y = ivy.native_array([0.1, 0.2, 0.1, 0.1, 0.5])
     >>> print(ivy.sparse_cross_entropy(x, y))
     ivy.array([0.693])

     With :class:`ivy.Container` input:

     >>> x = ivy.Container(a=ivy.array([4]))
     >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.1, 0.1, 0.5]))
     >>> print(ivy.sparse_cross_entropy(x, y))
     {
         a: ivy.array([0.693])
     }

     With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

     >>> x = ivy.array([0])
     >>> y = ivy.native_array([0.1, 0.2, 0.6, 0.1])
     >>> print(ivy.sparse_cross_entropy(x,y))
     ivy.array([2.3])

     With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

     >>> x = ivy.array([0])
     >>> y = ivy.Container(a=ivy.array([0.1, 0.2, 0.6, 0.1]))
     >>> print(ivy.sparse_cross_entropy(x,y))
     {
         a: ivy.array([2.3])
     }

     Instance Method Examples
     ------------------------

     With :class:`ivy.Array` input:

     >>> x = ivy.array([2])
     >>> y = ivy.array([0.1, 0.1, 0.7, 0.1])
     >>> print(x.sparse_cross_entropy(y))
     ivy.array([0.357])

     With :class:`ivy.Container` input:

     >>> x = ivy.Container(a=ivy.array([2]))
     >>> y = ivy.Container(a=ivy.array([0.1, 0.1, 0.7, 0.1]))
     >>> print(x.sparse_cross_entropy(y))
     {
         a: ivy.array([0.357])
     }

    """
    ivy.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    true = ivy.one_hot(true, pred.shape[axis])
    return ivy.cross_entropy(
        true, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
    )
