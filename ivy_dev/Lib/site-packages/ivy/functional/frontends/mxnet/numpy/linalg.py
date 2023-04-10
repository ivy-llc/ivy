import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    return ivy.tensordot(a, b, axes=axes)
