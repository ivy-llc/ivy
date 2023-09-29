# my_code.py

import ivy

@with_supported_dtypes({
    "2.5.1 and below": ("bool", "float16", "float32", "float64", "int32", "int64")
}, "paddle")
@to_ivy_arrays_and_back
def greater_equal(x, y, name=None):
    current_backend = ivy.current_backend(x, y)
    x = ivy.to_ivy(x)
    y = ivy.to_ivy(y)
    return ivy.current_backend(x, y).greater_equal(x, y)
