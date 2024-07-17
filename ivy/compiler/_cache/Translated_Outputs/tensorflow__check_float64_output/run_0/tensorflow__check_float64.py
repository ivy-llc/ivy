import math

from .tensorflow__helpers import tensorflow_dtype
from .tensorflow__helpers import tensorflow_is_array


def tensorflow__check_float64(input):
    if tensorflow_is_array(input):
        return tensorflow_dtype(input) == "float64"
    if math.isfinite(input):
        m, e = math.frexp(input)
        return abs(input) > 3.4028235e38 or e < -126 or e > 128
    return False
