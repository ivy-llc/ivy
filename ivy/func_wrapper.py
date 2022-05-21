import ivy
import inspect
import importlib
import numpy as np
from types import ModuleType


wrapped_modules_n_classes = []
NON_WRAPPED_FUNCTIONS = [
    "copy_nest",
    "current_framework",
    "current_framework_str",
    "set_framework",
    "get_framework",
    "unset_framework",
    "set_debug_mode",
    "set_breakpoint_debug_mode",
    "set_exception_debug_mode",
    "unset_debug_mode",
    "debug_mode",
    "nested_map",
    "to_ivy",
    "args_to_ivy",
    "to_native",
    "args_to_native",
    "default",
    "exists",
    "set_min_base",
    "get_min_base",
    "set_min_denominator",
    "get_min_denominator",
    "split_func_call_across_gpus",
    "cache_fn",
    "split_func_call",
    "compile",
    "compile_graph",
    "dev",
    "dev",
    "as_ivy_dev",
    "as_native_dev",
    "memory_on_dev",
    "gpu_is_available",
    "num_gpus",
    "tpu_is_available",
    "dtype",
    "as_ivy_dtype",
    "cprint",
    "to_ivy_module",
    "tree_flatten",
    "tree_unflatten",
    "start_compiling",
    "stop_compiling",
    "get_compiled",
    "index_nest",
    "set_nest_at_index",
    "map_nest_at_index",
    "multi_index_nest",
    "set_nest_at_indices",
    "map_nest_at_indices",
    "nested_indices_where",
    "map",
    "unset_default_device",
    "closest_valid_dtype",
    "default_dtype",
    "as_native_dtype",
    "is_ivy_array",
    "is_ivy_container",
    "inplace_update",
    "inplace_increment",
    "inplace_decrement",
    "prune_nest_at_index",
    "prune_nest_at_indices",
    "is_array",
    "is_native_array",
    "nested_any",
    "fn_array_spec",
    "insert_into_nest_at_index",
    "insert_into_nest_at_indices",
    "vec_sig_fig",
    "native_array",
]
FUNCTIONS_W_CONT_SUPPORT = [
    "multi_head_attention",
    "execute_with_gradients",
    "adam_step",
    "optimizer_update",
    "gradient_descent_update",
    "lars_update",
    "adam_update",
    "lamb_update",
    "stable_divide",
    "stable_pow",
]
ARRAYLESS_RET_FUNCTIONS = [
    "to_numpy",
    "to_list",
    "to_scalar",
    "is_native_array",
    "is_ivy_array",
    "is_variable",
]
NESTED_ARRAY_RET_FUNCTIONS = ["unstack", "split"]

FW_FN_KEYWORDS = {
    "numpy": [],
    "jax": [],
    "tensorflow": [],
    "torch": [],
    "mxnet": ["ndarray"],
}

NATIVE_KEYS_TO_SKIP = {
    "numpy": [],
    "jax": [],
    "tensorflow": [],
    "torch": [
        "classes",
        "torch",
        "is_grad_enabled",
        "get_default_dtype",
        "numel",
        "clone",
        "cpu",
        "set_",
        "type",
        "requires_grad_",
    ],
    "mxnet": [],
}


# Functions #


def _wrap_function(fn):
    """
    Creates a wrapped ivy version of the function if it is not a private function and
    not in the non wrapped functions list. This allows the new function to accept as
    inputs an ivy array before performing the required o  peration and then returning
    an ivy array.

    Parameters
    ----------
    fn
        native function to be wrapped

    Returns
    -------
        The wrapped version of the function with all the necessary attributes updated.
    """
    # do nothing if the function is private or in the non wrapped functions list
    if hasattr(fn, "__name__") and (
            fn.__name__[0] == "_" or fn.__name__ in NON_WRAPPED_FUNCTIONS
    ):
        return fn

    # do nothing if the function is already wrapped
    if hasattr(fn, "wrapped") and fn.wrapped:
        return fn

    def _function_w_arrays_handled(*args, out=None, **kwargs):
        native_args, native_kwargs = ivy.args_to_native(
            *args, **kwargs, include_derived={tuple: True}
        )

        """
        computes the result of the function fn, returning the result as an ivy array or
        a native framework array.

        Parameters
        ----------
        args
            The arguments to be passed to the function.
        
        out
            optional output array, for writing the result to.

        kwargs
            The key word arguments to be passed  to the function.
        
        Returns
        -------
            The result of computing the function fn as an ivy array or a native array.
        """
        if ivy.exists(out):
            native_out = ivy.to_native(out)
            native_or_ivy_ret = fn(*native_args, out=native_out, **native_kwargs)
        else:
            native_or_ivy_ret = fn(*native_args, **native_kwargs)
        if fn.__name__ in ARRAYLESS_RET_FUNCTIONS + NESTED_ARRAY_RET_FUNCTIONS:
            return native_or_ivy_ret
        elif ivy.exists(out) and ivy.is_ivy_array(out):
            out.data = ivy.to_native(native_or_ivy_ret)
            return out
        return ivy.to_ivy(native_or_ivy_ret, nested=True, include_derived={tuple: True})

    def _get_first_array(*args, **kwargs):
        # ToDo: make this more efficient, with function ivy.nested_nth_index_where
        arr = None
        if args:
            arr_idxs = ivy.nested_indices_where(args, ivy.is_array)
            if arr_idxs:
                arr = ivy.index_nest(args, arr_idxs[0])
            else:
                arr_idxs = ivy.nested_indices_where(kwargs, ivy.is_array)
                if arr_idxs:
                    arr = ivy.index_nest(kwargs, arr_idxs[0])
        elif kwargs:
            arr_idxs = ivy.nested_indices_where(kwargs, ivy.is_array)
            if arr_idxs:
                arr = ivy.index_nest(kwargs, arr_idxs[0])
        return arr

    def _function_w_arrays_dtype_n_dev_handled(*args, **kwargs):
        handle_dtype = 'dtype' in kwargs
        handle_dev = 'device' in kwargs
        if handle_dtype or handle_dev:
            arr = _get_first_array(*args, **kwargs)
            if handle_dtype:
                kwargs['dtype'] = ivy.default_dtype(kwargs['dtype'], item=arr, as_native=True)
            if handle_dev:
                kwargs['dev'] = ivy.default_device(kwargs['device'], item=arr, as_native=True)
        return _function_w_arrays_handled(*args, **kwargs)

    def _function_wrapped(*args, **kwargs):
        """
        Computes the result of the function fn, returning the result as an ivy array,
        a native framework array, or an ivy container.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The key word arguments to be passed to the function.

        Returns
        -------
            The result of computing the function fn as an ivy array, a native array,
            or an ivy container.
        """
        fn_name = fn.__name__
        """ 
        if the function is not implemented for containers or the function 
        has built-in container support, call the function using the passed 
        arguments directly, returning an ivy or a native array.
        """
        if not hasattr(ivy.Container, fn_name) or fn_name in FUNCTIONS_W_CONT_SUPPORT:
            return _function_w_arrays_dtype_n_dev_handled(*args, **kwargs)
        """
        if any of the arguments or keyword arguments passed to the function contains a 
        a container, get the container's version of the function and call it using
        the passed arguments.
        """
        if ivy.nested_any(
            args, ivy.is_ivy_container, check_nests=True
        ) or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True):
            if args and ivy.is_ivy_container(args[0]):
                f = getattr(ivy.Container, fn_name)
            else:
                f = getattr(ivy.StaticContainer, fn_name)
            if "out" in f.__code__.co_varnames:
                return f(*args, **kwargs)
            return f(*args, **kwargs)

        """
        if the passed arguments does not contain a container, the function using 
        the passed arguments, returning an ivy or a native array.
        """
        return _function_w_arrays_dtype_n_dev_handled(*args, **kwargs)

    if hasattr(fn, "__name__"):
        _function_wrapped.__name__ = fn.__name__
    _function_wrapped.wrapped = True
    _function_wrapped.inner_fn = fn
    if hasattr(fn, "array_spec"):
        _function_wrapped.array_spec = fn.array_spec
    if hasattr(fn, "reduce"):
        _function_wrapped.reduce = fn.reduce

    return _function_wrapped


def _unwrap_function(function_wrapped):
    """
    Unwraps the function in function_wrapped.

    Parameters
    ----------
    function_wrapped
        The function to be unwrapped.

    Returns
    -------
    The unwrapped version of the function which is the same as the passed function
    for unwrapped functions and the inner_fn if the function is wrapped. The newly unwrapped
    function accepts inputs and returns outputs as native arrays instead of ivy arrays.
    """
    if not hasattr(function_wrapped, "wrapped") or not function_wrapped.wrapped:
        return function_wrapped
    return function_wrapped.inner_fn


def _invalid_fn(fn, fs=None):
    if fs is None:
        fs = ivy.current_framework_str()
    if isinstance(fn, np.ufunc):
        return False
    if not hasattr(fn, "__module__") or not fn.__module__:
        return True
    fw_fn_keywords = ["ivy", fs] + FW_FN_KEYWORDS[fs]
    for kw in fw_fn_keywords:
        if kw in fn.__module__:
            return False
    return True


def _wrap_or_unwrap_functions(
    wrap_or_unwrap_fn, val=None, fs=None, classes_to_wrap=None, native=False, depth=0
):
    classes_to_wrap = [] if classes_to_wrap is None else classes_to_wrap
    if fs is None:
        fs = ivy.current_framework_str()
    if val is None:
        val = importlib.import_module(ivy.current_framework_str()) if native else ivy
    str_to_check = fs if native else "ivy"
    is_class = inspect.isclass(val)
    if isinstance(val, ModuleType) or (val in classes_to_wrap):
        if val in wrapped_modules_n_classes or (
            (
                "__file__" not in val.__dict__
                or (str_to_check not in val.__file__)
                or "framework_handler" in val.__file__
            )
            and not is_class
        ):
            return val
        wrapped_modules_n_classes.append(val)
        if is_class:
            for k in dir(val):
                if native and (k in NATIVE_KEYS_TO_SKIP[fs]):
                    continue
                v = getattr(val, k)
                if v is not None:
                    # noinspection PyBroadException
                    try:
                        setattr(
                            val,
                            k,
                            _wrap_or_unwrap_functions(
                                wrap_or_unwrap_fn,
                                v,
                                fs,
                                classes_to_wrap,
                                native,
                                depth + 1,
                            ),
                        )
                    except Exception:
                        pass
        else:
            for k, v in val.__dict__.items():
                if native and (k in NATIVE_KEYS_TO_SKIP[fs] or k[0] == "_"):
                    continue
                if v is None:
                    val.__dict__[k] = v
                else:
                    # noinspection PyBroadException
                    try:
                        val.__dict__[k] = _wrap_or_unwrap_functions(
                            wrap_or_unwrap_fn, v, fs, classes_to_wrap, native, depth + 1
                        )
                    except Exception:
                        pass
        if depth == 0:
            wrapped_modules_n_classes.clear()
        return val
    elif callable(val) and not is_class:
        if depth == 0:
            wrapped_modules_n_classes.clear()
        if (
            hasattr(val, "inner_fn") and (_invalid_fn(val.inner_fn) and not native)
        ) or (_invalid_fn(val) and not native):
            return val
        return wrap_or_unwrap_fn(val)
    if depth == 0:
        wrapped_modules_n_classes.clear()
    return val


def _wrap_functions():
    return _wrap_or_unwrap_functions(_wrap_function)


def _unwrap_functions():
    return _wrap_or_unwrap_functions(_unwrap_function)
