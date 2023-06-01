# local
import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
@with_supported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "paddle")
def binary_cross_entropy(input, label, weight=None, reduction="mean", name=None):
    result = ivy.binary_cross_entropy(label, input, epsilon=0.0, reduction=reduction)

    if weight is not None:
        result = ivy.multiply(weight, result)

    return result
