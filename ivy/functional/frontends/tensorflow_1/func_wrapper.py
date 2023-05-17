# global
import inspect
from typing import Callable
import functools

# local
import ivy
import ivy.functional.frontends.tensorflow_1 as frontend


def to_ivy_dtype(dtype):
    if not dtype or isinstance(dtype, str):
        return dtype
    return frontend.as_dtype(dtype).ivy_dtype


def handle_tf_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_tf_dtype(*args, dtype=None, **kwargs):
        if len(args) > (dtype_pos + 1):
            dtype = args[dtype_pos]
            kwargs = {
                **dict(
                    zip(
                        list(inspect.signature(fn).parameters.keys())[
                            dtype_pos + 1 : len(args)
                        ],
                        args[dtype_pos + 1 :],
                    )
                ),
                **kwargs,
            }
            args = args[:dtype_pos]
        elif len(args) == (dtype_pos + 1):
            dtype = args[dtype_pos]
            args = args[:-1]
        dtype = to_ivy_dtype(dtype)
        return fn(*args, dtype=dtype, **kwargs)

    dtype_pos = list(inspect.signature(fn).parameters).index("dtype")
    _handle_tf_dtype.handle_tf_dtype = True
    return _handle_tf_dtype


def _tf_frontend_array_to_ivy(x):
    if hasattr(x, "ivy_array"):
        return x.ivy_array
    return x


def _native_to_ivy_array(x):
    if isinstance(x, ivy.NativeArray):
        return ivy.array(x)
    return x


def _to_ivy_array(x):
    return _tf_frontend_array_to_ivy(_native_to_ivy_array(x))
