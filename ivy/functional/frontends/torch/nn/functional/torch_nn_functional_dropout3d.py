# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
@with_unsupported_dtypes({'1.11.0 and below': ('float16',)}, 'torch')
def dropout3d(input, p=0.5, training=True, inplace=False):
    input = ivy.cast(input, 'float32')
    if training:
        return ivy.dropout3d(input, p, inplace=inplace)
    else:
        return input
