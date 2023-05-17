# local
from ivy.functional.frontends.tensorflow_1.tensor import EagerTensor
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import handle_tf_dtype


@handle_tf_dtype
def constant(value, dtype=None, shape=None, name=None):
    if shape is not None:
        value = ivy.reshape(value, shape=shape)
    if dtype is not None:
        return EagerTensor(ivy.astype(value, dtype))
    return EagerTensor(value)
