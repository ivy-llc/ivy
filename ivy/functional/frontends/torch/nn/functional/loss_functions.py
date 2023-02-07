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
    if weight is not None:
        ret = ivy.multiply(weight, ret)
    ret = _apply_reduction(reduction, size_average, reduce, ret)
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def binary_cross_entropy(
    input, target, weight=None, size_average=None, reduce=None, reduction="mean"
):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.binary_cross_entropy(target, input, epsilon=0.0)

    if weight is not None:
        result = ivy.multiply(weight, result)
    result = reduction(result)
    return result


@to_ivy_arrays_and_back
def binary_cross_entropy_with_logits(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction="mean",
    pos_weight=None,
):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.binary_cross_entropy_with_logits(target, input, pos_weight=pos_weight)

    if weight is not None:
        result = ivy.multiply(weight, result)
    result = reduction(result).astype(target.dtype)
    return result


@to_ivy_arrays_and_back
def mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.square(input - target)
    result = reduction(result)
    return result


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
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

    if beta < 1e-5:
        # [Copied and modified from fvcore]
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * _diff_abs ** 2 / 0" has an incoming
        # gradient of zeros, rather than "no gradient"). To avoid this
        # issue, we define small values of beta to be exactly l1 loss.
        loss = ivy.abs(input - target)
    else:
        _diff_abs = ivy.abs(input - target)

        loss = ivy.where(
            _diff_abs < beta,
            0.5 * _diff_abs**2 / beta,
            _diff_abs - 0.5 * beta,
        )

    ret = reduction(loss)

    return ret


@to_ivy_arrays_and_back
def huber_loss(
    input,
    target,
    reduction="mean",
    delta=1.0,
):
    delta = ivy.array(delta)
    _diff_abs = ivy.abs(ivy.subtract(input, target))

    loss = ivy.where(
        _diff_abs < delta,  # If |xᵢ - yᵢ| < δ
        0.5 * _diff_abs**2,  # lᵢ = 0.5(xᵢ - yᵢ)²
        delta * (_diff_abs - 0.5 * delta),
    )  # lᵢ = δ(|xᵢ - yᵢ| - 0.5 * δ)

    reduction = _get_reduction(reduction)
    ret = reduction(loss)

    return ivy.astype(ret, input.dtype)


@to_ivy_arrays_and_back
def l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    loss = ivy.abs(input - target)
    reduction = _get_reduction(reduction, size_average, reduce)
    ret = reduction(loss)
    return ret


@to_ivy_arrays_and_back
def nll_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
):
    out = ivy.zeros_like(target)

    if len(input.shape) == 1:
        for i in range(len(target)):
            out[i] = input[target[i]]
    else:
        for i in range(len(target)):
            out[i] = input[i][target[i]]
    loss = -out

    if weight is not None:
        loss = ivy.multiply(weight, loss)
    reduct = _get_reduction(reduction, size_average, reduce)
    ret = reduct(loss)

    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def soft_margin_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    loss = ivy.log1p(ivy.exp(-input * target))
    reduction = _get_reduction(reduction, size_average, reduce)
    ret = reduction(loss)
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def kl_div(
    input, target, size_average=None, reduce=None, reduction="mean", log_target=False
):
    size = ivy.shape(input)

    if len(size) < 1:
        size = [1]

    def loss_fn():
        if log_target:
            return ivy.exp(target) * (target - input)
        return target * (ivy.log(target) - input)

    def batchmean(x):
        if not reduce:
            return x / size[0]

        if size_average:
            return ivy.mean(x) / size[0]

        return ivy.sum(x) / size[0]

    loss = ivy.nan_to_num(loss_fn())

    if reduction == "batchmean":
        reduction = batchmean
    else:
        reduction = _get_reduction(reduction, size_average, reduce)

    return reduction(loss)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
def margin_ranking_loss(
    input1,
    input2,
    target,
    margin=0.0,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    loss = -1 * target * (input1 - input2) + margin
    loss = ivy.where(loss < 0, 0, loss)
    reduction = _get_reduction(reduction, size_average, reduce)
    return reduction(loss)


@to_ivy_arrays_and_back
def ctc_loss(
    log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean"
):
    ivy.set_backend("torch")
    input_lengths = ivy.asarray(input_lengths)
    target_lengths = ivy.asarray(target_lengths)
    dt = log_probs.dtype
    log_probs = log_probs.double()
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0, dtype=ivy.int64)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[
                cum_target_length - target_length : cum_target_length
            ]
        probs = log_probs[:input_length, i].exp()
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = targets_prime[:-2] != targets_prime[2:]
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += ivy.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        # ==========================================================================================================
        losses.append(-alpha[-2:].sum().log()[None])
    output = ivy.frontends.torch.cat(losses, 0)
    if reduction == "mean":
        return (output / target_lengths).mean()
    elif reduction == "sum":
        return output.sum()
    output = output.to(dt)
    return output
