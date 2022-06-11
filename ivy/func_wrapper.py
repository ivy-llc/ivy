import ivy
import inspect
import functools
from types import FunctionType
from typing import Callable

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
    "set_default_dtype",
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


def inputs_to_native_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
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

    new_fn.inputs_to_native_arrays = True
    return new_fn


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
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

    new_fn.inputs_to_ivy_arrays = True
    return new_fn


def outputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
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

    new_fn.outputs_to_ivy_arrays = True
    return new_fn


def to_native_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.NativeArray` instances
    and return arrays are all converted to `ivy.Array` instances.
    """
    return outputs_to_ivy_arrays(inputs_to_native_arrays(fn))


# Data Type Handling #
# -------------------#


def infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
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

    new_fn.infer_dtype = True
    return new_fn


# Device Handling #
# ----------------#


def infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
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

    new_fn.infer_device = True
    return new_fn


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn: Callable) -> Callable:
    handle_out_in_backend = "out" in inspect.signature(fn).parameters.keys()

    @functools.wraps(fn)
    def new_fn(*args, out=None, **kwargs):
        """
        Calls `fn` with the `out` argument handled correctly for performing an inplace
        update.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        out
            The array to write the result to.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `out` handled correctly for
            inplace updates.
        """
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

    new_fn.handle_out_argument = True
    return new_fn


# Nestable Handling #
# ------------------#


def handle_nestable(fn: Callable) -> Callable:
    fn_name = fn.__name__
    cont_fn = getattr(ivy.Container, "static_" + fn_name)

    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls `fn` with the *nestable* property of the function correctly handled.
        This means mapping the function to the container leaves if any containers are
        passed in the input.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with the nestable property handled correctly.
        """
        # if any of the arguments or keyword arguments passed to the function contains
        # a container, get the container's version of the function and call it using
        # the passed arguments.
        if ivy.nested_any(
            args, ivy.is_ivy_container, check_nests=True
        ) or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    new_fn.handle_nestable = True
    return new_fn


# Functions #


def _wrap_function(value: Callable, original: Callable) -> Callable:
    """Apply wrapping to backend implementation `value` if the original implementation
    `original is also wrapped, and if `value` is not already wrapped. `handle_nestable`,
    `infer_device` etc are set during wrapping, hence indicate to us whether a certain
    function is wrapped or not.

    Parameters
    ----------
    value
        the backend implementation to potentially wrap
    original
        the original implementation of `value` which tells us which wrappers we need.

    Returns
    -------
    ret
        `value` appropriately wrapped if `value` is a function, otherwise just the
        input is returned.
    """
    if isinstance(value, FunctionType):
        if (
            hasattr(original, "handle_nestable")
            and not hasattr(value, "handle_nestable")
        ) or (
            hasattr(ivy.Container, value.__name__)
            and value.__name__ not in FUNCTIONS_W_CONT_SUPPORT + NON_WRAPPED_FUNCTIONS
        ):
            value = handle_nestable(value)
        if hasattr(original, "infer_device") and not hasattr(value, "infer_device"):
            value = infer_device(value)
        if hasattr(original, "infer_dtype") and not hasattr(value, "infer_dtype"):
            value = infer_dtype(value)
        if hasattr(original, "handle_out_argument") and not hasattr(
            value, "handle_out_argument"
        ):
            value = handle_out_argument(value)
        if hasattr(original, "inputs_to_native_arrays") and not hasattr(
            value, "inputs_to_native_arrays"
        ):
            value = inputs_to_native_arrays(value)
        if hasattr(original, "outputs_to_ivy_arrays") and not hasattr(
            value, "outputs_to_ivy_arrays"
        ):
            value = outputs_to_ivy_arrays(value)
    return value
