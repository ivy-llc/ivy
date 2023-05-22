# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
def dropout(input, p=0.5, training=True, inplace=False):
    if not training:
        ret = input
    else:
        ret = ivy.dropout(input, p)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret
