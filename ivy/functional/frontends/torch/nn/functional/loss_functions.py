# global
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes


# --- Helpers --- #
# --------------- #


def _apply_reduction(reduction, size_average, reduce, to_reduce):
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
    return _get_reduction_method(reduction, to_reduce)


def _get_reduction(reduction, size_average=None, reduce=None):
    if size_average is not None or reduce is not None:
        return _get_reduction_func(_get_reduction_string(size_average, reduce))
    else:
        return _get_reduction_func(reduction)


def _get_reduction_func(reduction):
    if reduction == "none":

        def ret(x):
            return x

    elif reduction == "mean":
        ret = ivy.mean
    elif reduction == "sum":
        ret = ivy.sum
    else:
        raise ivy.utils.exceptions.IvyException(
            f"{reduction} is not a valid value for reduction"
        )
    return ret


def _get_reduction_method(reduction, to_reduce):
    if reduction == "none":
        ret = to_reduce
    elif reduction == "mean":
        ret = ivy.mean(to_reduce)
    elif reduction == "sum":
        ret = ivy.sum(to_reduce)
    else:
        raise ivy.utils.exceptions.IvyException(
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


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def binary_cross_entropy(
    input, target, weight=None, size_average=None, reduce=None, reduction="mean"
):
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
    result = ivy.binary_cross_entropy(target, input, epsilon=0.0, reduction=reduction)

    if weight is not None:
        result = ivy.multiply(weight, result)

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
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
    result = ivy.binary_cross_entropy(
        target,
        input,
        reduction=reduction,
        from_logits=True,
        pos_weight=pos_weight,
    )

    if weight is not None:
        result = ivy.multiply(weight, result)

    return result


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def cosine_embedding_loss(
    input1, input2, target, margin=0.0, size_average=None, reduce=None, reduction="mean"
):
    def norm(input, axis):
        return ivy.sqrt(ivy.sum(ivy.square(input), axis=axis))

    def cosine_similarity(x1, x2):
        axis = None
        if len(x1.shape) == len(x2.shape) and len(x2.shape) == 2:
            axis = 1
        input1_norm = norm(x1, axis=axis)
        input2_norm = norm(x2, axis=axis)
        norm_mm = input1_norm * input2_norm
        norm_mm, eps = torch_frontend.promote_types_of_torch_inputs(norm_mm, 1e-08)
        return ivy.sum(x1 * x2, axis=axis) / ivy.maximum(norm_mm, eps)

    def calculate_loss(x1, x2, target):
        cos = cosine_similarity(x1, x2)
        if target == ivy.array(1.0):
            loss = 1.0 - cos
        elif target == ivy.array(-1.0):
            loss = ivy.maximum(ivy.array(0.0), cos - ivy.array(margin))
        else:
            _, zero = torch_frontend.promote_types_of_torch_inputs(
                input1, ivy.array(0.0)
            )
            return zero

        return loss

    ivy.utils.assertions.check_true(
        target.ndim + 1 == input1.ndim and target.ndim + 1 == input2.ndim,
        f"{target.ndim}D target tensor expects {target.ndim + 1}D input tensors, but "
        f"found inputs with sizes {list(input1.shape)} and {list(input2.shape)}.",
    )

    ivy.utils.assertions.check_true(
        target.ndim < 2, "0D or 1D target tensor expected, multi-target not supported"
    )

    ivy.utils.assertions.check_shape(input1, input2)

    if target.ndim == 1:
        ivy.utils.assertions.check_true(
            target.shape[0] == input1.shape[0],
            f"The size of target tensor ({target.shape[0]}) must match the size of"
            f" input tensor ({input1.shape[0]}) at non-singleton dimension 0 ",
        )

    if target.ndim == 0:
        loss = calculate_loss(input1, input2, target)
    else:
        loss = ivy.array(
            [
                calculate_loss(input1[i], input2[i], target[i])
                for i in range(input1.shape[0])
            ]
        )

    reduction = _get_reduction(reduction, size_average, reduce)
    loss = reduction(loss)
    return loss


def cosine_similarity(x1, x2):
    axis = None
    if len(x1.shape) == len(x2.shape) and len(x2.shape) == 2:
        axis = 1
    input1_norm = norm(x1, axis=axis)
    input2_norm = norm(x2, axis=axis)
    norm_mm = input1_norm * input2_norm
    norm_mm, eps = torch_frontend.promote_types_of_torch_inputs(norm_mm, 1e-08)
    return ivy.sum(x1 * x2, axis=axis) / ivy.maximum(norm_mm, eps)


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
    loss = ivy.cross_entropy(target, input, epsilon=label_smoothing, reduction="none")

    if ignore_index != -100:
        mask = ivy.not_equal(target, ignore_index)
        loss = ivy.where(mask, loss, ivy.zeros_like(loss))

    if weight is not None:
        result = ivy.multiply(weight, loss)

    reduction = _get_reduction(reduction, size_average, reduce)
    return reduction(result).astype(target.dtype)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("bool", "integer")}, "torch")
def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction="mean"):
    input, target = torch_frontend.promote_types_of_torch_inputs(input, target)
    target, var = torch_frontend.promote_types_of_torch_inputs(target, var)
    if var.shape != input.shape:
        if input.shape[:-1] == var.shape:
            var = torch_frontend.unsqueeze(var, dim=2)
        elif input.shape[:-1] == var.shape[:-1] and var.shape[-1] == 1:
            pass
        else:
            raise ivy.utils.exceptions.IvyError("var is of incorrect size")

    if reduction is not None and reduction != "mean" and reduction != "sum":
        raise ivy.utils.exceptions.IvyError(f"{reduction} is not valid")

    if ivy.any(var < 0):
        raise ivy.utils.exceptions.IvyError("var has negative entry/entries")

    var = ivy.maximum(var, eps)

    loss = 0.5 * (ivy.log(var) + (input - target) ** 2 / var)

    if full:
        loss += 0.5 * ivy.log(2 * ivy.pi)

    reduction = _get_reduction_func(reduction)
    ret = reduction(loss)

    return ret.astype(input.dtype)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
@to_ivy_arrays_and_back
def hinge_embedding_loss(
    input,
    target,
    margin=1.0,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    margin = ivy.array(margin)

    loss = ivy.where(
        ivy.logical_or(target == -1, target == 1),
        ivy.where(target == 1, input, ivy.maximum(0, margin - input)),
        ivy.maximum(margin, input),
    )

    reduction = _get_reduction(reduction, size_average, reduce)
    ret = reduction(loss)

    return ivy.astype(ret, input.dtype)


@to_ivy_arrays_and_back
def huber_loss(
    input,
    target,
    reduction="mean",
    delta=1.0,
):
    return ivy.huber_loss(target, input, delta=delta, reduction=reduction)


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
def kl_div(
    input, target, size_average=None, reduce=None, reduction="mean", log_target=False
):
    orig_red = reduction
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
    else:
        reduction = reduction if reduction != "batchmean" else "sum"
    ret = ivy.kl_div(input, target, reduction=reduction, log_target=log_target)
    if orig_red == "batchmean" and input.ndim != 0:
        ret = ret / input.shape[0]
    return ret


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.2 and below": ("float", "complex")}, "torch")
def l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    if size_average is not None or reduce is not None:
        reduction = _get_reduction_string(size_average, reduce)
    ret = ivy.l1_loss(input, target, reduction=reduction)
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def margin_ranking_loss(
    input1,
    input2,
    target,
    margin=0.0,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    input1, input2 = torch_frontend.promote_types_of_torch_inputs(input1, input2)
    input2, target = torch_frontend.promote_types_of_torch_inputs(input2, target)
    loss = -1 * target * (input1 - input2) + margin
    loss = ivy.where(loss < 0, 0, loss)
    reduction = _get_reduction(reduction, size_average, reduce)
    return reduction(loss).astype(input1.dtype)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
def mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    reduction = _get_reduction(reduction, size_average, reduce)
    result = ivy.square(input - target)
    result = reduction(result)
    return result


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def multilabel_margin_loss(
    input, target, size_average=None, reduce=None, reduction="mean"
):
    ivy.assertions.check_true(
        input.shape == target.shape,
        lambda: (
            "Same shape is expected for both output and target, but instead got :"
            f" output {input.shape} and target : {target.shape}"
        ),
    )
    input, target = torch_frontend.promote_types_of_torch_inputs(input, target)
    pos = input[ivy.astype(target, bool)]
    neg = input[ivy.astype(1 - target, bool)]
    loss = ivy.maximum(0, 1 - (torch_frontend.unsqueeze(pos, dim=1) - neg))
    reduct = _get_reduction(reduction, size_average, reduce)
    return reduct(loss)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def multilabel_soft_margin_loss(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    loss = -(
        target * ivy.log(ivy.sigmoid(input))
        + (1 - target) * ivy.log(1 - ivy.sigmoid(input))
    )

    if weight is not None:
        loss = ivy.multiply(weight, loss)

    class_dim = ivy.get_num_dims(input) - 1
    C = ivy.shape(input)[class_dim]

    loss = ivy.sum(loss, axis=class_dim) / C

    reduction = _get_reduction(reduction, size_average, reduce)
    ret = reduction(loss)

    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.2 and below": ("float16", "int8", "int16", "int32")}, "torch"
)
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


def norm(input, axis):
    return ivy.sqrt(ivy.sum(ivy.square(input), axis=axis))


def pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
    x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)
    x1_dim = len(x1.shape)
    x2_dim = len(x2.shape)
    if x1_dim > x2_dim:
        output_dim = x1_dim
    else:
        output_dim = x2_dim

    return ivy.vector_norm(x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def poisson_nll_loss(
    input,
    target,
    log_input=True,
    full=False,
    size_average=None,
    eps=1e-8,
    reduce=None,
    reduction="mean",
):
    input, target = torch_frontend.promote_types_of_torch_inputs(input, target)
    if log_input:
        loss = ivy.exp(input) - target * input
    else:
        loss = input - target * ivy.log(input + eps)
    if full:
        approximation = (
            target * ivy.log(target) - target + 0.5 * ivy.log(2 * ivy.pi * target)
        )
        loss += ivy.where(target > 1, approximation, 0)

    reduction = _get_reduction(reduction, size_average, reduce)
    return reduction(loss).astype(input.dtype)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def smooth_l1_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
    beta=1.0,
):
    return ivy.smooth_l1_loss(input, target, beta=beta, reduction=reduction)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def soft_margin_loss(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    return ivy.soft_margin_loss(input, target, reduction=reduction)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-06,
    swap=False,
    size_average=None,
    reduce=None,
    reduction="mean",
):
    def pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
        x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)
        x1_dim = len(x1.shape)
        x2_dim = len(x2.shape)
        if x1_dim > x2_dim:
            output_dim = x1_dim
        else:
            output_dim = x2_dim

        return ivy.vector_norm(
            x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim
        )

    reduction = _get_reduction(reduction, size_average, reduce)

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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
def triplet_margin_with_distance_loss(
    anchor,
    positive,
    negative,
    distance_function=None,
    margin=1.0,
    swap=False,
    reduction="mean",
):
    reduction = _get_reduction(reduction)

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

    if distance_function is None:
        distance_function = pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = ivy.minimum(dist_neg, dist_swap)

    loss = ivy.maximum(dist_pos - dist_neg + ivy.array(margin), ivy.array(0.0))

    return reduction(loss).astype(anchor.dtype)
