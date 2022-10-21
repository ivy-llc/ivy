import ivy

# hann_ window

def hann_window(window_length, periodic=True, dtype=ivy.int32, name=None):
    return ivy.hann_window(window_length, periodic, dtype, name)


