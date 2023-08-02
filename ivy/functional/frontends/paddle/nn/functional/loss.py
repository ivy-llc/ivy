# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
import ivy.functional.frontends.paddle as paddle
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.frontends.paddle.func_wrapper import (
    inputs_to_ivy_arrays,
    to_ivy_arrays_and_back,
)


# helpers
def _get_reduction_func(reduction):
    if reduction == "none":
        ret = lambda x: x
    elif reduction == "mean":
        ret = ivy.mean
    elif reduction == "sum":
        ret = ivy.sum
    else:
        raise ivy.utils.exceptions.IvyException(
            "{} is not a valid value for reduction".format(reduction)
        )
    return ret


@with_supported_dtypes(
    {"2.5.1 and below": ("float32",)},
    "paddle",
)
@inputs_to_ivy_arrays
def binary_cross_entropy_with_logits(
    logit,
    label,
    weight=None,
    reduction="mean",
    pos_weight=None,
    name=None,
):
    ret = ivy.binary_cross_entropy(
        label, logit, from_logits=True, reduction="none", pos_weight=pos_weight
    )
    reduction = _get_reduction_func(reduction)
    if weight is not None:
        ret = ivy.multiply(weight, ret)
    ret = reduction(ret).astype(label.dtype)
    return paddle.to_tensor(ivy.atleast_1d(ret))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@inputs_to_ivy_arrays
def mse_loss(input, label, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)
    ret = ivy.square(input - label)
    ret = reduction(ret)

    if ret.shape == ():
        ret = ret.expand_dims()

    return paddle.to_tensor(ret)


@handle_exceptions
@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def cosine_embedding_loss(
    input1, input2, label, margin=0.0, reduction="mean", name=None
):
    if len(label.shape) != 1:
        raise ValueError("1D target tensor expected, multi-target not supported")

    if input1.shape != input2.shape:
        raise ValueError(
            "the shape of input tensor 1 should be equal to input tensor 2, but found"
            " inputs with different sizes"
        )

    if len(input1.shape) > 2:
        raise ValueError(
            "1D target tensor expects 1D or 2D input tensors, but found inputs with"
            " different sizes"
        )

    prod_sum = (input1 * input2).sum(axis=-1)
    mag_square1 = ivy.square(input1).sum(axis=-1) + 1e-11
    mag_square2 = ivy.square(input2).sum(axis=-1) + 1e-11
    denom = ivy.sqrt(mag_square1 * mag_square2)
    cos = prod_sum / denom
    zeros = ivy.zeros_like(cos)
    pos = 1 - cos
    neg = ivy.clip(cos - margin, 0, 0)
    out_pos = ivy.where(label == 1, pos, zeros)
    out_neg = ivy.where(label == -1, neg, zeros)
    out = out_pos + out_neg

    if reduction == "none":
        pass
    if reduction == "mean":
        out = ivy.mean(out)
    elif reduction == "sum":
        out = ivy.sum(out)

    return out


@with_supported_dtypes(
    {"2.5.1 and below": ("float32",)},
    "paddle",
)
@to_ivy_arrays_and_back
def hinge_embedding_loss(input, label, margin=1.0, reduction="mean"):
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "'reduction' in 'hinge_embedding_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction)
        )

    zero_ = ivy.zeros([1], dtype=input.dtype)
    loss = ivy.where(label == 1.0, input, zero_) + ivy.where(
        label == -1.0, ivy.functional.ivy.activations.relu(margin - input), zero_
    )

    if reduction == "mean":
        return ivy.mean(loss)
    elif reduction == "sum":
        return ivy.sum(loss)
    elif reduction == "none":
        return loss


@with_supported_dtypes(
    {"2.5.1 and below": ("float32",)},
    "paddle",
)
@to_ivy_arrays_and_back
def log_loss(input, label, epsilon=0.0001, name=None):
    out = -label * ivy.log(input + epsilon) - (
        (1 - label) * ivy.log(1 - input + epsilon)
    )
    return out


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def smooth_l1_loss(
    input,
    label,
    reduction="mean",
    delta=1.0,
    name=None,
):
    sum_diff = ivy.abs(input - label).astype(label.dtype)
    condition = sum_diff <= delta
    out = ivy.where(
        condition,
        0.5 * ivy.pow(ivy.abs(input - label), 2).astype(label.dtype),
        (delta * ivy.abs(ivy.abs(input - label))).astype(label.dtype)
        - (0.5 * ivy.pow(delta, 2)).astype(label.dtype),
    )
    if reduction == "none":
        pass
    elif reduction == "mean":
        out = ivy.mean(out)
    elif reduction == "sum":
        out = ivy.sum(out)
    return out.astype(label.dtype)


@inputs_to_ivy_arrays
def l1_loss(
    input,
    label,
    reduction="mean",
    name=None,
):
    sum_diff = ivy.abs(input - label)
    reduction = _get_reduction_func(reduction)
    out = reduction(sum_diff)
    if out.shape == ():
        out = out.expand_dims()
    return paddle.to_tensor(out)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def kl_div(
    input,
    label,
    reduction="mean",
    name=None,
):
    if input.shape != label.shape:
        raise ValueError(
            "the shape of input tensor should be equal to target tensor, but found"
            " inputs with different sizes"
        )

    out = label * (ivy.log(label) - input)

    size = ivy.shape(input)
    if len(size) < 1:
        size = [1]

    if reduction == "mean":
        out = ivy.mean(out)
    elif reduction == "batchmean":
        out = ivy.sum(out) / size[0]
    elif reduction == "sum":
        out = ivy.sum(out)
    else:
        pass
    return out.astype(label.dtype)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def margin_ranking_loss(input, other, label, margin=0.0, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)

    out = ivy.subtract(input, other)
    neg_label = ivy.negative(label)
    out = ivy.multiply(neg_label, out)

    if margin != 0.0:
        margin_var = ivy.full([1], margin, dtype=out.dtype)
        out = ivy.add(out, margin_var)

    out = ivy.where(out < 0, 0, out)
    out = reduction(out).astype(input.dtype)
    out = ivy.atleast_1d(out)

    return out


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-06,
    swap=False,
    reduction="mean",
):
    def pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
        x1, x2 = paddle.promote_types_of_paddle_inputs(x1, x2)
        x1_dim = len(x1.shape)
        x2_dim = len(x2.shape)
        if x1_dim > x2_dim:
            output_dim = x1_dim
        else:
            output_dim = x2_dim

        return ivy.vector_norm(
            x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim
        )

    reduction = _get_reduction_func(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim

    ivy.assertions.check_true(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            "The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    dist_positive = pairwise_distance(anchor, positive, p=p, eps=eps)
    dist_negative = pairwise_distance(anchor, negative, p=p, eps=eps)
    if swap:
        dist_swap = pairwise_distance(positive, negative, p=p, eps=eps)
        dist_negative = ivy.minimum(dist_negative, dist_swap)
    loss = ivy.maximum(
        dist_positive - dist_negative + ivy.array(margin), ivy.array(0.0)
    )

    loss = reduction(loss).astype(anchor.dtype)
    return loss
