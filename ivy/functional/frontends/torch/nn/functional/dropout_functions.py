# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def dropout(input, p=0.5, training=True, inplace=False):
    ret = ivy.dropout(input, p, training=training)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def dropout2d(tensor, prob=0.5, training=True, inplace=False):
    if tensor.ndim < 2:
        raise ValueError("Feature dropout requires at least 2 dimensions in the input")

    if not training:
        return tensor

    ret = tensor.dropout2d(prob)

    if inplace:
        ivy.inplace_update(tensor, ret)
        return tensor
    return ret
