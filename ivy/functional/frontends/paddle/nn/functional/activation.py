# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)

# local
from ivy.functional.frontends.paddle.tensor.math import tanh as paddle_tanh
tanh = paddle_tanh

@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def relu(x, name=None):
    return ivy.relu(x)

@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sigmoid(x, name=None):
    return ivy.sigmoid(x)

@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def leaky_relu(x, alpha=0.2, name=None):
    return ivy.leaky_relu(x, alpha=alpha)
