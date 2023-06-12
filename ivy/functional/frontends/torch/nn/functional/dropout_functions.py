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
def dropout2d(tensor, p=0.5, training=True, inplace=False):
    if tensor.ndim != 4:
        raise ValueError("Input must have exactly 4 dimensions.")

    if not training:
        return tensor

    shape = tensor.shape
    mask = ivy.bernoulli(probs=p, shape=(shape[2], shape[3]))
    mask = ivy.tile(mask, shape[1])
    mask = ivy.reshape(mask, (shape[1], shape[2], shape[3]))
    mask = ivy.tile(mask, shape[0])
    mask = ivy.reshape(mask, shape)

    if inplace:
        tensor *= mask
        return tensor
    else:
        return tensor * mask
