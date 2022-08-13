import ivy
import functools
from types import FunctionType
from typing import Callable

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
        arr_idxs = ivy.nested_indices_where(args, ivy.is_array, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(args, arr_idxs[0])
        else:
            arr_idxs = ivy.nested_indices_where(
                kwargs, ivy.is_array, stop_after_n_found=1
            )
            if arr_idxs:
                arr = ivy.index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = ivy.nested_indices_where(kwargs, ivy.is_array, stop_after_n_found=1)
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
        if not ivy.get_array_mode():
            return fn(*args, **kwargs)
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.NativeArray instances
        new_args, new_kwargs = ivy.args_to_native(
            *args, **kwargs, include_derived={tuple: True}
        )
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

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
        if not ivy.get_array_mode():
            return ret
        # convert all arrays in the return to `ivy.Array` instances
        return ivy.to_ivy(ret, nested=True, include_derived={tuple: True})

    new_fn.outputs_to_ivy_arrays = True
    return new_fn


def from_zero_dim_arrays_to_float(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
        Calls the function, and then converts all 0 dimensional array instances in
        the function to float numbers if out argument is not provided.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with 0 dimensional arrays as float numbers.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)
        # get out arg index
        out_arg_pos = ivy.arg_info(fn, name="out")["idx"]
        # check if out is None or out is not present in args and kwargs.
        out_args = out_arg_pos < len(args) and args[out_arg_pos] is None
        out_kwargs = "out" in kwargs and kwargs["out"] is None
        if ret.shape == () and (out_args or out_kwargs):
            return float(ret)
        # convert to float from 0 dim
        return ret

    new_fn.zero_dim_arrays_to_float = True
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
        dtype = ivy.default_dtype(dtype=dtype, item=arr, as_native=True)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    new_fn.infer_dtype = True
    return new_fn


def integer_array_to_float(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """
         Promotes all the integer array inputs passed to the function both
         as positional or keyword arguments to the default float dtype.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with integer array arguments
            promoted to default float dtype.

        """

        if args:

            args = list(args)
            for i in range(len(args)):
                if ivy.is_array(args[i]):
                    if ivy.is_int_dtype(args[i].dtype):
                        is_native = ivy.is_native_array(args[i])
                        args[i] = ivy.asarray(args[i], dtype=ivy.default_float_dtype())
                        if is_native:
                            args[i] = ivy.to_native(args[i])

            args = tuple(args)
        if kwargs:
            for i in kwargs:
                if ivy.is_array(kwargs[i]):
                    if ivy.is_int_dtype(kwargs[i].dtype):
                        is_native = ivy.is_native_array(kwargs[i])
                        kwargs[i] = ivy.asarray(
                            kwargs[i], dtype=ivy.default_float_dtype()
                        )
                        if is_native:
                            kwargs[i] = ivy.to_native(kwargs[i])
        return fn(*args, **kwargs)

    new_fn.integer_array_to_float = True
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
    handle_out_in_backend = hasattr(fn, "support_native_out")

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
        cont_fn = getattr(ivy.Container, "static_" + fn_name)
        if ivy.get_nestable_mode() and (
            ivy.nested_any(args, ivy.is_ivy_container, check_nests=True)
            or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True)
        ):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    new_fn.handle_nestable = True
    return new_fn


# Functions #


def _wrap_function(key: str, to_wrap: Callable, original: Callable) -> Callable:
    """Apply wrapping to backend implementation `to_wrap` if the original implementation
    `original` is also wrapped, and if `to_wrap` is not already wrapped. Attributes
    `handle_nestable`, `infer_device` etc are set during wrapping, hence indicate to
    us whether a certain function has been wrapped or not. Also handles wrapping of the
    `linalg` namespace.

    Parameters
    ----------
    to_wrap
        the new implementation to potentially wrap
    original
        the original implementation of `to_wrap` which tells us which wrappers we need.

    Returns
    -------
    ret
        `to_wrap` appropriately wrapped if `to_wrap` is a function, otherwise just the
        input is returned.
    """
    if key == "linalg":
        for linalg_k, linalg_v in to_wrap.__dict__.items():
            if (
                isinstance(linalg_v, FunctionType)
                and linalg_k != "namedtuple"
                and not linalg_k.startswith("_")
            ):
                to_wrap.__dict__[linalg_k] = _wrap_function(
                    linalg_k, linalg_v, ivy.__dict__[linalg_k]
                )
        return to_wrap
    if isinstance(to_wrap, FunctionType):
        # set attributes
        for attr in original.__dict__.keys():
            # private attribute or decorator
            if attr.startswith("_") or hasattr(ivy, attr) or attr == "handles_out_arg":
                continue
            setattr(to_wrap, attr, getattr(original, attr))
        # wrap decorators (sequence matters)
        for attr in [
            "infer_device",
            "infer_dtype",
            "integer_array_to_float",
            "outputs_to_ivy_arrays",
            "inputs_to_native_arrays",
            "inputs_to_ivy_arrays",
            "handle_out_argument",
            "handle_nestable",
        ]:
            if hasattr(original, attr) and not hasattr(to_wrap, attr):
                to_wrap = getattr(ivy, attr)(to_wrap)
    return to_wrap
