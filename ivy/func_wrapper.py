import ivy
import inspect
import importlib
import numpy as np
from types import ModuleType
from typing import Callable, Optional, List, Union


wrapped_modules_n_classes = []
NON_WRAPPED_FUNCTIONS = [
    "copy_nest",
    "current_backend",
    "current_backend_str",
    "set_backend",
    "get_backend",
    "unset_backend",
    "get_referrers_recursive",
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
    "set_default_device",
    "unset_default_device",
    "closest_valid_dtype",
    "default_dtype",
    "default_device",
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
NON_DTYPE_WRAPPED_FUNCTIONS = [
    "arange",
    "asarray",
    "array",
    "full",
    "prod",
    "sum",
    "astype",
]
NON_DEV_WRAPPED_FUNCTIONS = []

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

# Helpers #
# --------#


# noinspection DuplicatedCode
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


# Array Handling #
# ---------------#


def inputs_to_native_arrays(fn):
    def new_fn(*args, **kwargs):
        """
        Converts all `ivy.Array` instances in both the positional and keyword arguments
        into `ivy.NativeArray` instances, and then calls the function with the updated
        arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with native arrays passed in the arguments.
        """
        # convert all arrays in the inputs to ivy.NativeArray instances
        native_args, native_kwargs = ivy.args_to_native(
            *args, **kwargs, include_derived={tuple: True}
        )
        return fn(*native_args, **native_kwargs)

    return new_fn


def inputs_to_ivy_arrays(fn):
    def new_fn(*args, **kwargs):
        """
        Converts all `ivy.NativeArray` instances in both the positional and keyword
        arguments into `ivy.Array` instances, and then calls the function with the
        updated arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays passed in the arguments.
        """
        # convert all arrays in the inputs to ivy.Array instances
        ivy_args, ivy_kwargs = ivy.args_to_ivy(
            *args, **kwargs, include_derived={tuple: True}
        )
        return fn(*ivy_args, **ivy_kwargs)

    return new_fn


def outputs_to_ivy_arrays(fn):
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all `ivy.NativeArray` instances in
        the function return into `ivy.Array` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with native arrays as ivy arrays.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)
        # convert all arrays in the return to `ivy.Array` instances
        return ivy.to_ivy(ret, nested=True, include_derived={tuple: True})

    return new_fn


def to_native_arrays_and_back(fn):
    return outputs_to_ivy_arrays(inputs_to_native_arrays(fn))


# Data Type Handling #
# -------------------#


def infer_dtype(fn):
    def new_fn(*args, dtype=None, **kwargs):
        """
        Determines the correct `dtype`, and then calls the function with the `dtype`
        passed explicitly.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        dtype
            The data type for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `dtype` passed explicitly.
        """
        # find the first array argument, if required
        arr = None if ivy.exists(dtype) else _get_first_array(*args, **kwargs)
        # infer the correct data type
        dtype = ivy.default_dtype(dtype, item=arr, as_native=True)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    return new_fn


# Device Handling #
# ----------------#


def infer_device(fn):
    def new_fn(*args, device=None, **kwargs):
        """
        Determines the correct `device`, and then calls the function with the `device`
        passed explicitly.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        device
            The device for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `device` passed explicitly.
        """
        # find the first array argument, if required
        arr = None if ivy.exists(device) else _get_first_array(*args, **kwargs)
        # infer the correct device
        device = ivy.default_device(device, item=arr, as_native=True)
        # call the function with device provided explicitly
        return fn(*args, device=device, **kwargs)

    return new_fn


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn):
    handle_out_in_backend = "out" in inspect.signature(fn).parameters.keys()

    def new_fn(*args, out=None, **kwargs):
        if out is None:
            return fn(*args, **kwargs)
        if handle_out_in_backend:
            # extract underlying native array for out
            native_out = ivy.to_native(out)
            # compute return, with backend inplace update handled by
            # the backend function
            ret = fn(*args, out=native_out, **kwargs)
            out.data = ivy.to_native(ret)
            return out
        # compute return, and then handle the inplace update explicitly
        ret = fn(*args, **kwargs)
        return ivy.inplace_update(out, ret)

    return new_fn


# Functions #


def _wrap_function(function: Callable) -> Callable:
    """
    Creates a wrapped ivy version of the function if it is not a private function and
    not in the non wrapped functions list. This allows the new function to accept as
    inputs an ivy array before performing the required operation and then returning
    an ivy array.

    Parameters
    ----------
    function
        native function to be wrapped

    Returns
    -------
        The wrapped version of the function with all the necessary attributes updated.

    Examples
    --------
    This gives us the original `abs` implementation:

    >>> from ivy.functional.backends.jax import abs as ivy_jax_abs
    >>> print(ivy_jax_abs)
    <function abs at 0x7fef19ddf040>

    whereas the following gives us the wrapped version of `abs`:

    >>> from ivy.functional.backends.jax import abs as ivy_jax_abs
    >>> from ivy.func_wrapper import _wrap_function
    >>> abs_wrapped = _wrap_function(ivy_jax_abs)
    >>> print(abs_wrapped)
    <function _wrap_function.<locals>._function_wrapped at 0x7fef1852a0d0>

    """
    # determine whether the function has an out argument
    keys = inspect.signature(function).parameters.keys()
    handle_out_with_backend = "out" in keys
    handle_dtype = "dtype" in keys
    handle_dev = "device" in keys

    # do nothing if the function is private or in the non wrapped functions list
    if hasattr(function, "__name__") and (
        function.__name__[0] == "_" or function.__name__ in NON_WRAPPED_FUNCTIONS
    ):
        return function

    # do nothing if the function is already wrapped
    if hasattr(function, "wrapped") and function.wrapped:
        return function

    def _function_w_arrays_n_out_handled(*args, out=None, **kwargs):
        """
        Converts all `ivy.Array` instances in both the positional and keyword arguments
        into `ivy.NativeArray` instances, calls the internal function `function`, and
        then converts all `ivy.NativeArray` instances in the return back to `ivy.Array`
        instances. Also handles `out` argument correctly, enabling an inplace update.

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
        # convert all arrays in the inputs to ivy.NativeArray instances
        native_args, native_kwargs = ivy.args_to_native(
            *args, **kwargs, include_derived={tuple: True}
        )
        if ivy.exists(out):
            # extract underlying native array for out
            native_out = ivy.to_native(out)
            if handle_out_with_backend:
                # compute return, with backend inplace update handled by
                # the backend function
                ret = function(*native_args, out=native_out, **native_kwargs)
            else:
                # compute return, with backend inplace update handled explicitly
                ret = function(*native_args, **native_kwargs)
                ret = ivy.inplace_update(native_out, ivy.to_native(ret))
        else:
            ret = function(*native_args, **native_kwargs)
        if function.__name__ in ARRAYLESS_RET_FUNCTIONS + NESTED_ARRAY_RET_FUNCTIONS:
            return ret
        elif ivy.exists(out):
            # handle ivy.Array inplace update as well
            out.data = ivy.to_native(ret)
            return out
        # convert all returned arrays to ivy.Array instances
        return ivy.to_ivy(ret, nested=True, include_derived={tuple: True})

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

    def _function_w_arrays_dtype_n_dev_handled(
        *args, dtype=None, device=None, **kwargs
    ):
        if handle_dtype or handle_dev:
            arr = _get_first_array(*args, **kwargs)
            if handle_dtype:
                if function.__name__ not in NON_DTYPE_WRAPPED_FUNCTIONS:
                    dtype = ivy.default_dtype(dtype, item=arr, as_native=True)
                kwargs["dtype"] = dtype
            if handle_dev:
                if function.__name__ not in NON_DEV_WRAPPED_FUNCTIONS:
                    device = ivy.default_device(device, item=arr, as_native=True)
                kwargs["device"] = device
        return _function_w_arrays_n_out_handled(*args, **kwargs)

    def _function_wrapped(*args, **kwargs):
        """
        Computes the result of `function`, returning the result as an ivy array,
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
        fn_name = function.__name__
        # if the function is not implemented for containers or the function has built-in
        # container support, call the function using the passed arguments directly,
        # returning an ivy or a native array.
        if not hasattr(ivy.Container, fn_name) or fn_name in FUNCTIONS_W_CONT_SUPPORT:
            return _function_w_arrays_dtype_n_dev_handled(*args, **kwargs)
        # if any of the arguments or keyword arguments passed to the function contains a
        # a container, get the container's version of the function and call it using
        # the passed arguments.
        if ivy.nested_any(
            args, ivy.is_ivy_container, check_nests=True
        ) or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True):
            f = getattr(ivy.Container, "static_" + fn_name)
            return f(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return _function_w_arrays_dtype_n_dev_handled(*args, **kwargs)

    if hasattr(function, "__name__"):
        _function_wrapped.__name__ = function.__name__
    _function_wrapped.wrapped = True
    _function_wrapped.inner_fn = function
    if hasattr(function, "array_spec"):
        _function_wrapped.array_spec = function.array_spec
    if hasattr(function, "reduce"):
        _function_wrapped.reduce = function.reduce

    return _function_wrapped


def _unwrap_function(function_wrapped: Callable) -> Callable:
    """
    Unwraps the function `function_wrapped`.

    Parameters
    ----------
    function_wrapped
        The function to be unwrapped.

    Returns
    -------
    The unwrapped version of the function which is the same as the passed function
    for unwrapped functions and the inner_fn if the function is wrapped.
    The newly unwrapped function accepts inputs and returns outputs as native arrays
    instead of ivy arrays.
    """
    if not hasattr(function_wrapped, "wrapped") or not function_wrapped.wrapped:
        return function_wrapped
    return function_wrapped.inner_fn


def _invalid_function(function: Callable, framework: Optional[str] = None) -> bool:
    if framework is None:
        framework = ivy.current_backend_str()
    if isinstance(function, np.ufunc):
        return False
    if not hasattr(function, "__module__") or not function.__module__:
        return True
    fw_fn_keywords = ["ivy", framework] + FW_FN_KEYWORDS[framework]
    for kw in fw_fn_keywords:
        if kw in function.__module__:
            return False
    return True


def _wrap_or_unwrap_functions(
    wrap_or_unwrap_function: Callable,
    val: Optional[Union[ModuleType, Callable]] = None,
    framework: Optional[str] = None,
    classes_to_wrap: Optional[List] = [],
    native: Optional[bool] = False,
    depth: Optional[int] = 0,
) -> Union[Callable, ModuleType]:
    if framework is None:
        framework = ivy.current_backend_str()
    if val is None:
        val = importlib.import_module(ivy.current_backend_str()) if native else ivy
    str_to_check = framework if native else "ivy"
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
        # if `val` is a class we recursively call `_wrap_or_unwrap_functions`
        # on every member of the class
        if is_class:
            for k in dir(val):
                if native and (k in NATIVE_KEYS_TO_SKIP[framework]):
                    continue
                v = getattr(val, k)
                if v is not None:
                    # noinspection PyBroadException
                    try:
                        setattr(
                            val,
                            k,
                            _wrap_or_unwrap_functions(
                                wrap_or_unwrap_function,
                                v,
                                framework,
                                classes_to_wrap,
                                native,
                                depth + 1,
                            ),
                        )
                    except Exception:
                        pass
        # or if `val` is a module, we recursively call
        # `_wrap_or_unwrap_functions` on each value of its dict
        else:
            for k, v in val.__dict__.items():
                if native and (k in NATIVE_KEYS_TO_SKIP[framework] or k[0] == "_"):
                    continue
                if v is None:
                    val.__dict__[k] = v
                else:
                    # noinspection PyBroadException
                    try:
                        val.__dict__[k] = _wrap_or_unwrap_functions(
                            wrap_or_unwrap_function,
                            v,
                            framework,
                            classes_to_wrap,
                            native,
                            depth + 1,
                        )
                    except Exception:
                        pass
        if depth == 0:
            wrapped_modules_n_classes.clear()
        return val
    # if `val` is a function/method we wrap it and return it (unless
    # there are issues with it being an invalid function)
    elif callable(val) and not is_class:
        if depth == 0:
            wrapped_modules_n_classes.clear()
        if (
            hasattr(val, "inner_fn")
            and (_invalid_function(val.inner_fn) and not native)
        ) or (_invalid_function(val) and not native):
            return val
        return wrap_or_unwrap_function(val)
    if depth == 0:
        wrapped_modules_n_classes.clear()
    return val


def _wrap_functions():
    return _wrap_or_unwrap_functions(_wrap_function)


def _unwrap_functions():
    return _wrap_or_unwrap_functions(_unwrap_function)
