import numpy as np


from .tensorflow__helpers import tensorflow__check_float64
from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_imag_1
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_real_1


def tensorflow__check_complex128(input):
    if tensorflow_is_array(input):
        return tensorflow_dtype(input) == "complex128"
    elif isinstance(input, np.ndarray):
        return str(input.dtype) == "complex128"
    if hasattr(input, "real") and hasattr(input, "imag"):
        return tensorflow__check_float64(
            tensorflow_real_1(input)
        ) and tensorflow__check_float64(tensorflow_imag_1(input))
    return False
