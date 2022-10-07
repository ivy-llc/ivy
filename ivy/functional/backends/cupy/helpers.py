import functools
from typing import Callable

import cupy as cp

def _handle_0_dim_output(function: Callable) -> Callable:
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return cp.asarray(ret) if not isinstance(ret, cp.ndarray) else ret

    return new_function
