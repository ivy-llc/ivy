from .func_wrapper import use_keras_backend_framework


# Array API Standard #
# -------------------#


@use_keras_backend_framework
def astype(x, dtype, /, *, copy=True, out=None):
    pass


@use_keras_backend_framework
def broadcast_arrays(*arrays):
    pass


@use_keras_backend_framework
def broadcast_to(x, /, shape, *, out=None):
    pass


@use_keras_backend_framework
def finfo(type, /):
    pass


@use_keras_backend_framework
def iinfo(type, /):
    pass


@use_keras_backend_framework
def result_type(*arrays_and_dtypes):
    pass


# Extra #
# ------#


@use_keras_backend_framework
def as_ivy_dtype(dtype_in, /):
    pass


@use_keras_backend_framework
def as_native_dtype(dtype_in):
    pass


@use_keras_backend_framework
def dtype(x, *, as_native=False):
    pass


@use_keras_backend_framework
def dtype_bits(dtype_in, /):
    pass


@use_keras_backend_framework
def is_native_dtype(dtype_in, /):
    pass
