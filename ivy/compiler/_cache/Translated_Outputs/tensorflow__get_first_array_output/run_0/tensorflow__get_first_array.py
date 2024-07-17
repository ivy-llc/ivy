from .tensorflow__helpers import tensorflow_index_nest
from .tensorflow__helpers import tensorflow_is_array
from .tensorflow__helpers import tensorflow_nested_argwhere


def tensorflow__get_first_array(*args, **kwargs):
    def array_fn(x):
        return (
            tensorflow_is_array(x)
            if not hasattr(x, "_ivy_array")
            else tensorflow_is_array(x.ivy_array)
        )

    array_fn = array_fn if "array_fn" not in kwargs else kwargs["array_fn"]
    arr = None
    if args:
        arr_idxs = tensorflow_nested_argwhere(args, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = tensorflow_index_nest(args, arr_idxs[0])
        else:
            arr_idxs = tensorflow_nested_argwhere(
                kwargs, array_fn, stop_after_n_found=1
            )
            if arr_idxs:
                arr = tensorflow_index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = tensorflow_nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = tensorflow_index_nest(kwargs, arr_idxs[0])
    return arr
