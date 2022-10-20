import ivy.functional.frontends.tensorflow as ivy_tf
# hann_window

def hann_window(window_length, periodic=True, dtype=ivy_ty.int32, name=None):
    return ivy.hann_window(window_length, periodic, dtype)

