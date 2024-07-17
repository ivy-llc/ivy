import functools
from typing import Callable

from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_to_ivy


def tensorflow__asarray_to_native_arrays_and_back(fn: Callable):
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back_wrapper(*args, dtype=None, **kwargs):
        new_arg = args[0]
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = tensorflow_default_dtype(dtype=dtype, as_native=True)
        return tensorflow_to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    _asarray_to_native_arrays_and_back_wrapper._asarray_to_native_arrays_and_back = True
    return _asarray_to_native_arrays_and_back_wrapper
