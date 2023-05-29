# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes


def _get_reduction_func(reduction):
    if reduction == "none":
        ret = lambda x: x
    elif reduction == "mean":
        ret = ivy.mean
    elif reduction == "sum":
        ret = ivy.sum
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def binary_cross_entropy(input, label, weight=None, reduction="mean", name=None):
    reduction_fn = _get_reduction_func(reduction)
    result = ivy.binary_cross_entropy(label, input, epsilon=0.0)

    if weight is not None:
        result = ivy.multiply(weight, result)

    if reduction in ["sum", "mean"]:
        result = reduction_fn(result).reshape((1,))
    else:
        result = reduction_fn(result)
    return result
