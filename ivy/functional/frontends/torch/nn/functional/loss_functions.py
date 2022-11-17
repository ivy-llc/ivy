# global
import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


def _get_reduction_func(reduction):
    if reduction == "none":
        ret = lambda x: x
    elif reduction == "mean":
        ret = ivy.mean
    elif reduction == "sum":
        ret = ivy.sum
    else:
        raise ivy.exceptions.IvyException(
            "{} is not a valid value for reduction".format(reduction)
        )
    return ret


def _legacy_get_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


def _get_reduction(reduction, size_average=None, reduce=None):
    if size_average is not None or reduce is not None:
        return _get_reduction_func(_legacy_get_string(size_average, reduce))
    else:
        return _get_reduction_func(reduction)


def _get_reduction_method(reduction, to_reduce):
    if reduction == "none":
        ret = to_reduce
    elif reduction == "mean":
        ret = ivy.mean(to_reduce)
    elif reduction == "sum":
        ret = ivy.sum(to_reduce)
    else:
        raise ivy.exceptions.IvyException(
            f"{reduction} is not a valid value for reduction"
        )
    return ret


def _get_reduction_string(size_average, reduce):
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


def _apply_reduction(reduction, size_average, reduce, to_reduce):
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
        return _get_reduction_method(reduction, to_reduce)
    else:
        return _get_reduction_method(reduction, to_reduce)


@to_ivy_arrays_and_back
def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    input = ivy.softmax(input)
    ret = ivy.cross_entropy(target, input, epsilon=label_smoothing)
	@@ -97,7 +97,7 @@ def cross_entropy(
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def binary_cross_entropy(
    input, target, weight=None, size_average=None, reduce=None, reduction="mean"
):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.binary_cross_entropy(target, input, epsilon=0.0)
	@@ -118,12 +118,12 @@ def mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):

@to_ivy_arrays_and_back
def smooth_l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
    beta=1.0,
):
    beta = ivy.array(beta, device=input.device)
    reduction = _get_reduction(reduction, size_average, reduce)
	@@ -141,7 +141,7 @@ def smooth_l1_loss(

        loss = ivy.where(
            _diff_abs < beta,
            0.5 * _diff_abs**2 / beta,
            _diff_abs - 0.5 * beta,
        )

	@@ -150,13 +150,46 @@ def smooth_l1_loss(
    return ret

@to_ivy_arrays_and_back
def huber_loss(
        input,
        target,
        reduction="mean",
        delta=1.0
):
    # Broadcast delta
    delta = ivy.array(delta, device=input.device)

    if delta == 1.0:  # If δ == 1, use smooth_l1_loss
        return smooth_l1_loss(input, target, reduction)

    else:  # Otherwise
        # Let's define the absolute diff |xᵢ - yᵢ| as _abs_diff
        _abs_diff = ivy.abs(input - target)

        loss = ivy.where(
            _abs_diff < delta,  # If |xᵢ - yᵢ| < δ
            0.5 * _abs_diff ** 2,  # lᵢ = 0.5(xᵢ - yᵢ)²
            delta * (_abs_diff - 0.5 * delta),  # l_i = δ(|xᵢ - yᵢ| - 0.5 * δ)
        )

        # Get the reduction from its string
        reduction = _get_reduction(reduction)

        # Apply reduction on loss
        loss = reduction(loss)

        # Return the loss
        return loss


@ to_ivy_arrays_and_back
def l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    loss=ivy.abs(input - target)
    reduction=_get_reduction(reduction, size_average, reduce)