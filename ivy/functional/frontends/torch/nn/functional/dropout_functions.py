# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import versions


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, versions["torch"])
def dropout(input, p=0.5, training=True, inplace=False):
    if not training:
        ret = input
    else:
        ret = ivy.dropout(input, p)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret
