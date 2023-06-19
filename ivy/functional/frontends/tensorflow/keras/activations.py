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
    return ivy.array(x)


@to_ivy_arrays_and_back
def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    return ivy.relu(x)


@to_ivy_arrays_and_back
def tanh(x):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def sigmoid(x):
    return ivy.sigmoid(x)


@to_ivy_arrays_and_back
def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)


@to_ivy_arrays_and_back
def gelu(x, approximate=False):
    return ivy.gelu(x, approximate=approximate)


@to_ivy_arrays_and_back
def softplus(x):
    return ivy.softplus(x)


@to_ivy_arrays_and_back
def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


@to_ivy_arrays_and_back
def swish(x):
    return ivy.multiply(x, ivy.sigmoid(x))


@to_ivy_arrays_and_back
def elu(x, alpha=1.0):
    zeros = ivy.zeros_like(x, dtype=ivy.dtype(x))
    ones = ivy.ones_like(x, dtype=ivy.dtype(x))
    alpha = ivy.astype(ivy.array(alpha), ivy.dtype(x))
    ret_val = ivy.where(
        x > zeros, x, ivy.multiply(alpha, ivy.subtract(ivy.exp(x), ones))
    )
    return ret_val


elu.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "bfloat16",
        "float32",
        "float64",
    ),
    "jax": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
}


@to_ivy_arrays_and_back
def selu(x):
    return ivy.selu(x)


selu.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "float32",
        "float64",
    ),
    "jax": (
        "float16",
        "float32",
        "float64",
    ),
}


def deserialize(name, custom_objects=None):
    return ivy.deserialize(name, custom_objects=custom_objects)


deserialize.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "float32",
        "float64",
    ),
    "jax": (
        "float16",
        "float32",
        "float64",
    ),
}


def get(name, custom_objects=None):
    return ivy.get(name, custom_objects=custom_objects)
