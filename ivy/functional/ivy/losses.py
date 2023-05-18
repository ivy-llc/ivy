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
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute cross-entropy between predicted and true discrete distributions.

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
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    pred = ivy.clip(pred, epsilon, 1 - epsilon)
    log_pred = ivy.log(pred)
    return _reduce_loss(reduction, log_pred * true, axis, out)


# @handle_exceptions
# @handle_nestable
# @handle_array_like_without_promotion
# @inputs_to_ivy_arrays
# @handle_array_function
# def binary_cross_entropy(
#     true: Union[ivy.Array, ivy.NativeArray],
#     pred: Union[ivy.Array, ivy.NativeArray],
#     /,
#     *,
#     epsilon: float = 1e-7,
#     reduction: str = "none",
#     out: Optional[ivy.Array] = None,
# ) -> ivy.Array:
#     """
#     Compute the binary cross entropy loss.

#     Parameters
#     ----------
#     true
#         input array containing true labels.
#     pred
#         input array containing Predicted labels.
#     epsilon
#         a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
#         loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
#     out
#         optional output array, for writing the result to. It must have a shape
#         that the inputs broadcast to.

#     Returns
#     -------
#     ret
#         The binary cross entropy between the given distributions.


#     Functional Examples
#     -------------------

#     With :class:`ivy.Array` input:

#     >>> x = ivy.array([0, 1, 0, 0])
#     >>> y = ivy.array([0.2, 0.8, 0.3, 0.8])
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     ivy.array([0.223,0.223,0.357,1.61])

#     >>> x = ivy.array([[0, 1, 0, 0]])
#     >>> y = ivy.array([[0.6, 0.2, 0.7, 0.3]])
#     >>> z = ivy.binary_cross_entropy(x, y, epsilon=1e-3)
#     >>> print(z)
#     ivy.array([[0.916,1.61,1.2,0.357]])

#     With :class:`ivy.NativeArray` input:

#     >>> x = ivy.native_array([0, 1, 0, 1])
#     >>> y = ivy.native_array([0.2, 0.7, 0.2, 0.6])
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     ivy.array([0.223,0.357,0.223,0.511])

#     With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

#     >>> x = ivy.array([0, 0, 1, 1])
#     >>> y = ivy.native_array([0.1, 0.2, 0.8, 0.6])
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     ivy.array([0.105,0.223,0.223,0.511])

#     With :class:`ivy.Container` input:

#     >>> x = ivy.Container(a=ivy.array([1, 0, 0]),b=ivy.array([0, 0, 1]))
#     >>> y = ivy.Container(a=ivy.array([0.6, 0.2, 0.3]),b=ivy.array([0.8, 0.2, 0.2]))
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     {a:ivy.array([0.511,0.223,0.357]),b:ivy.array([1.61,0.223,1.61])}

#     With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

#     >>> x = ivy.array([1 , 1, 0])
#     >>> y = ivy.Container(a=ivy.array([0.7, 0.8, 0.2]))
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     {
#        a: ivy.array([0.357, 0.223, 0.223])
#     }

#     Instance Method Examples
#     ------------------------

#     Using :class:`ivy.Array` instance method:

#     >>> x = ivy.array([1, 0, 0, 0])
#     >>> y = ivy.array([0.8, 0.2, 0.2, 0.2])
#     >>> z = ivy.binary_cross_entropy(x, y)
#     >>> print(z)
#     ivy.array([0.223, 0.223, 0.223, 0.223])
#     """
#     ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
#     pred = ivy.clip(pred, epsilon, 1 - epsilon)
#     return _reduce_loss(
#         reduction,
#         ivy.add(ivy.log(pred) * true, ivy.log(1 - pred) * (1 - true)),
#         None,
#         out,
#     )


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
    from_logits: bool = False,
    epsilon: float = 0.0,
    reduction: str = "none",
    pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    true = true.astype(pred.dtype)

    true = true.astype(pred.dtype)
    epsilon = ivy.asarray(epsilon, dtype=pred.dtype)

    true = true * (1.0 - epsilon) + 0.5 * epsilon

    if from_logits:
        if pos_weight is not None:
            if pos_weight.shape[0] != pred.shape[1]:
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
    reduction: str = "sum",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute sparse cross entropy between logits and labels.

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
    ivy.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    true = ivy.one_hot(true, pred.shape[axis])
    return ivy.cross_entropy(
        true, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
    )
