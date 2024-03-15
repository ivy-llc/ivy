import ivy
import ivy.functional.frontends.tensorflow as tf_frontend
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy import with_supported_dtypes


ACTIVATION_FUNCTIONS = [
    "gelu",
    "leaky_relu",
    "log_softmax",
    "relu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
]


# --- Helpers --- #
# --------------- #


# note: defined to avoid AST call extraction of
# 'tf_frontend.keras.activations.__dict__.items()
# or 'tf_frontend.keras.activations.__dict__.values()'
def _get_tf_keras_activations():
    return tf_frontend.keras.activations.__dict__.items()


# --- Main --- #
# ------------ #


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
def deserialize(name, custom_objects=None):
    if name is None:
        return None

    elif isinstance(name, str):
        if custom_objects and name in custom_objects:
            return custom_objects.get(name)

        # To replicate tensorflow framework
        elif (
            ivy.current_backend().__name__.split(".")[-1] == "tensorflow"
            and name in tf_frontend.keras.activations.__dict__
        ):  # noqa
            return tf_frontend.keras.activations.__dict__[name]

        # On other backends, query the function from global ivy dict
        elif name in ACTIVATION_FUNCTIONS:
            return ivy.__dict__[name]

        else:
            raise ValueError(f"Unknown activation function: {name}.")

    else:
        raise ValueError(f"Could not interpret activation function: {name}")


@with_supported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float16", "float32", "float64")},
    "tensorflow",
)
@to_ivy_arrays_and_back
def elu(x, alpha=1.0):
    zeros = ivy.zeros_like(x, dtype=ivy.dtype(x))
    ones = ivy.ones_like(x, dtype=ivy.dtype(x))
    alpha = ivy.astype(ivy.array(alpha), ivy.dtype(x))
    ret_val = ivy.where(
        x > zeros, x, ivy.multiply(alpha, ivy.subtract(ivy.exp(x), ones))
    )
    return ret_val


@to_ivy_arrays_and_back
def gelu(x, approximate=False):
    return ivy.gelu(x, approximate=approximate)


def get(identifier):
    if identifier is None:
        return tf_frontend.keras.activations.linear

    elif isinstance(identifier, str):
        return tf_frontend.keras.activations.deserialize(identifier)

    elif callable(identifier):
        return identifier

    else:
        raise ValueError(f"Could not interpret function identifier: {identifier}")


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


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
@to_ivy_arrays_and_back
def selu(x):
    return ivy.selu(x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
def serialize(activation, use_legacy_format=False, custom_objects=None):
    # If the activation function is None, return None
    if activation is None:
        return None

    # If the activation function is already a string, return it
    elif isinstance(activation, str):
        return activation

    # If the activation function is callable (a function), get its name
    elif callable(activation):
        # Check if the function is in the custom_objects dictionary
        if custom_objects:
            for name, custom_func in custom_objects.items():
                if custom_func == activation:
                    return name

        tf_keras_frontend_activations = _get_tf_keras_activations()

        # Check if the function is in the ACTIVATION_FUNCTIONS list
        if activation.__name__ in ACTIVATION_FUNCTIONS:
            return activation.__name__

        # Check if the function is in the TensorFlow frontend activations
        elif activation in [fn for name, fn in tf_keras_frontend_activations]:
            for name, tf_func in tf_keras_frontend_activations:
                if tf_func == activation:
                    return name

        else:
            raise ValueError(f"Unknown activation function: {activation}.")

    else:
        raise ValueError(f"Could not interpret activation function: {activation}")


@to_ivy_arrays_and_back
def sigmoid(x):
    return ivy.sigmoid(x)


@to_ivy_arrays_and_back
def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)


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
def tanh(x):
    return ivy.tanh(x)
