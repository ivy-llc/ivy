import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)


@handle_tf_dtype
@to_ivy_arrays_and_back
def kaiser_window(window_length, beta=12.0, dtype=ivy.float32, name=None):
    return ivy.kaiser_window(window_length, periodic=False, beta=beta, dtype=dtype)


kaiser_window.supported_dtypes = ("float32", "float64", "float16", "bfloat16")


# dct
@to_ivy_arrays_and_back
def dct(input, type=2, n=None, axis=-1, norm=None, name=None):
    return ivy.dct(input, type=type, n=n, axis=axis, norm=norm)


# vorbis_window
@to_ivy_arrays_and_back
def vorbis_window(window_length, dtype=ivy.float32, name=None):
    return ivy.vorbis_window(window_length, dtype=dtype, out=None)


# idct
@to_ivy_arrays_and_back
def idct(input, type=2, n=None, axis=-1, norm=None, name=None):
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return ivy.dct(input, type=inverse_type, n=n, axis=axis, norm=norm)
