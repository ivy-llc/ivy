import functools
from typing import Callable

import numpy as np


def _handle_0_dim_output(function: Callable) -> Callable:
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return np.asarray(ret) if not isinstance(ret, np.ndarray) else ret

    return new_function
