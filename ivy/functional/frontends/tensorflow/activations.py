import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def hard_sigmoid(x):
    dtype_in = x.dtype
    point_two = ivy.full(x.shape, 0.2)
    point_five = ivy.full(x.shape, 0.5)
    x = ivy.multiply(x, point_two)
    x = ivy.add(x, point_five)
    x = ivy.clip(x, 0.0, 1.0)
    x = ivy.asarray(x, dtype=dtype_in)
    return x

@to_ivy_arrays_and_back
def linear(x):
    return x

@to_ivy_arrays_and_back
def relu(x):
    return ivy.relu(x)

@to_ivy_arrays_and_back
def sigmoid(x):
    return ivy.sigmoid(x)

@to_ivy_arrays_and_back
def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)
