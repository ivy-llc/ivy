import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def hann_window(window_length, periodic=True, dtype=ivy.float32):
    return ivy.hann_window(window_length, periodic=periodic, dtype=dtype)


hann_window.supported_dtypes = ("float32", "float64", "float16", "bfloat16")
