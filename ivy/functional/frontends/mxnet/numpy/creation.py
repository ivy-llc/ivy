import ivy
from ivy.functional.frontends.mxnet.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@handle_numpy_dtype
@to_ivy_arrays_and_back
def array(object, dtype=None, ctx=None):
    if not ivy.is_array(object) and not dtype:
        return ivy.array(object, dtype="float32", device=ctx)
    return ivy.array(object, dtype=dtype, device=ctx)
