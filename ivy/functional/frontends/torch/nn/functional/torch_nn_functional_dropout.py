
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")

def dropout(input, p=0.5, training=True, inplace=False):
    if p<0 or p>1:
        raise ValueError("dropout prob has to be between 0 and 1")
    if training:
       val = ivy.dropout(input, p)
    else :
       val = input
    if inplace:
       return input

    return val
