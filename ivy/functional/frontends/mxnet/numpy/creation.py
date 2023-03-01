import ivy

from ivy.functional.frontends.mxnet.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def array(object, dtype=None, ctx=None):
    return ivy.array(object, dtype=dtype)
