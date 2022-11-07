import functools
from typing import Callable
import numpy as np


def scalar_output_to_0d_array(function: Callable) -> Callable:
    """
    Sometimes NumPy functions return scalars e.g. `np.add` does when
    the inputs are both 0 dimensional. We use this wrapper to handle such
    cases, and convert scalar outputs to 0d arrays, since the array API
    standard dictates outputs must be arrays.
    """

    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return np.asarray(ret) if not isinstance(ret, np.ndarray) else ret

    return new_function
