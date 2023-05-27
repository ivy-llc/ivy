# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


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


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
def binary_cross_entropy(input, label, weight=None, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)
    result = ivy.binary_cross_entropy(label, input, epsilon=0.0)

    if weight is not None:
        result = ivy.multiply(weight, result)
    result = reduction(result)
    return result
