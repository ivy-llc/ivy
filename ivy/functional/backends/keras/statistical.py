import ivy
from .func_wrapper import use_keras_backend_framework


# Array API Standard #
# -------------------#


@use_keras_backend_framework
def min(x, /, *, axis=None, keepdims=False, initial=None, where=None, out=None):
    pass

@use_keras_backend_framework
def max(x, /, *, axis=None, keepdims=False, out=None):
    pass


@use_keras_backend_framework
def mean(x, /, axis=None, keepdims=False, *, dtype=None, out=None):
    pass


@use_keras_backend_framework
def prod(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    pass


@use_keras_backend_framework
def std(x, /, *, axis=None, correction=0.0, keepdims=False, out=None):
    pass


@use_keras_backend_framework
def sum(x, /, *, axis=None, dtype=None, keepdims=False, out=None):
    pass


@use_keras_backend_framework
def var(x, /, *, axis=None, correction=0.0, keepdims=False, out=None):
    pass


# Extra #
# ------#


@use_keras_backend_framework
def cumprod(x, /, *, axis=0, exclusive=False, reverse=False, dtype=None, out=None):
    pass


@use_keras_backend_framework
def cumsum(x, axis=0, exclusive=False, reverse=False, *, dtype=None, out=None):
    pass


@use_keras_backend_framework
def einsum(equation, *operands, out=None):
    pass
