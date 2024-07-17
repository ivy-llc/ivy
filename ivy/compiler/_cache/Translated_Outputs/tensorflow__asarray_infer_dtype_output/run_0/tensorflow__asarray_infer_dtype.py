import tensorflow as tf
import numpy as np

import functools
from typing import Callable

from .tensorflow__helpers import tensorflow__flatten_nest
from .tensorflow__helpers import tensorflow_as_native_dtype
from .tensorflow__helpers import tensorflow_default_dtype
from .tensorflow__helpers import tensorflow_default_float_dtype
from .tensorflow__helpers import tensorflow_exists
from .tensorflow__helpers import tensorflow_nested_map
from .tensorflow__helpers import tensorflow_promote_types


def tensorflow__asarray_infer_dtype(fn: Callable):
    @functools.wraps(fn)
    def _asarray_infer_dtype_wrapper(*args, dtype=None, **kwargs):
        def _infer_dtype(obj):
            if isinstance(obj, tf.TensorShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return tensorflow_default_dtype(item=obj)

        if not tensorflow_exists(dtype):
            arr = args[0]
            dtype_list = [
                tensorflow_nested_map(lambda x: _infer_dtype(x), arr, shallow=False)
            ]
            dtype_list = tensorflow__flatten_nest(dtype_list)
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:
                dtype = dtype_list[0]
                for dt in dtype_list[1:]:
                    dtype = tensorflow_promote_types(dtype, dt)
            else:
                dtype = tensorflow_default_float_dtype()
            dtype = tensorflow_as_native_dtype(dtype)
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype_wrapper.infer_dtype = True
    return _asarray_infer_dtype_wrapper
