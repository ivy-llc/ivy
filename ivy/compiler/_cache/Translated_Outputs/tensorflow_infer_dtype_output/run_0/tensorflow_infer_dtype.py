import functools
from typing import Callable

from .tensorflow__helpers import tensorflow__get_first_array
from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_exists


def tensorflow_infer_dtype(fn: Callable):
    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
        arr = (
            None
            if tensorflow_exists(dtype)
            else tensorflow__get_first_array(*args, **kwargs)
        )
        dtype = tensorflow_default_dtype(dtype=dtype, item=arr, as_native=True)
        return fn(*args, dtype=dtype, **kwargs)

    _infer_dtype.infer_dtype = True
    return _infer_dtype
