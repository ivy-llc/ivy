import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    return ivy.softmax(data, axis=axis)
