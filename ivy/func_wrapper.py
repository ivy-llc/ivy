import contextlib
import ivy
import functools
import logging
import weakref
import warnings
import copy as python_copy
from types import FunctionType
from typing import Callable, Literal
import inspect
import numpy as np

from ivy.utils.exceptions import IvyValueError


# for wrapping (sequence matters)
FN_DECORATORS = [
    "handle_complex_input",
    "infer_device",
    "handle_device_shifting",
    "infer_dtype",
    "handle_array_function",
    "outputs_to_ivy_arrays",
    "outputs_to_ivy_shapes",
    "outputs_to_native_arrays",
    "inputs_to_native_arrays",
    "inputs_to_native_shapes",
    "inputs_to_ivy_arrays",
    "handle_out_argument",
    "handle_view_indexing",
    "handle_view",
    "handle_array_like_without_promotion",
    "handle_partial_mixed_function",
    "handle_nestable",
    "handle_ragged",
    "handle_exceptions",
    "handle_nans",
]


# Helpers #
# --------#

# for casting modes, order is the hierarchy
casting_modes_dict = {
    "uint": lambda: ivy.valid_uint_dtypes,
    "int": lambda: sorted(
        tuple(set(ivy.valid_int_dtypes).difference(set(ivy.valid_uint_dtypes)))
    ),
    "float": lambda: ivy.valid_float_dtypes,
    "complex": lambda: ivy.valid_complex_dtypes,
}


def caster(dtype, intersect):
    if hasattr(dtype, "dtype"):
        dtype = ivy.as_ivy_dtype(dtype.dtype)
    else:
        dtype = ivy.as_ivy_dtype(dtype)
    if str(dtype) in intersect:
        # based on upcasting or downcasting do something
        if ivy.cast_dtypes():
            # all casting types is enabled
            # check cross_casting
            ret_dtype = cross_caster(intersect)
            if ret_dtype:
                return ret_dtype
            # check upcasting
            ret_dtype = upcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
            # check downcasting
            ret_dtype = downcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.crosscast_dtypes:
            # check cross_casting
            ret_dtype = cross_caster(intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.upcast_dtypes:
            # check upcasting
            ret_dtype = upcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.downcast_dtypes:
            # check downcasting
            ret_dtype = downcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype


def upcaster(dtype, intersect):
    # upcasting is enabled, we upcast to the highest
    if "uint" in str(dtype):
        index = casting_modes_dict["uint"]().index(dtype) + 1
        result = ""
        while index < len(casting_modes_dict["uint"]()):
            if casting_modes_dict["uint"]()[index] not in intersect:
                result = casting_modes_dict["uint"]()[index]
                break
            index += 1
        return result

    if "int" in dtype:
        index = casting_modes_dict["int"]().index(dtype) + 1
        result = ""
        while index < len(casting_modes_dict["int"]()):
            if casting_modes_dict["int"]()[index] not in intersect:
                result = casting_modes_dict["int"]()[index]
                break
            index += 1
        return result

    if "float" in dtype:
        index = casting_modes_dict["float"]().index(dtype) + 1
        result = ""
        while index < len(casting_modes_dict["float"]()):
            if casting_modes_dict["float"]()[index] not in intersect:
                result = casting_modes_dict["float"]()[index]
                break
            index += 1
        return result

    if "complex" in dtype:
        index = casting_modes_dict["complex"]().index(dtype) + 1
        result = ""
        while index < len(casting_modes_dict["complex"]()):
            if casting_modes_dict["complex"]()[index] not in intersect:
                result = casting_modes_dict["complex"]()[index]
                break
            index += 1
        return result


def downcaster(dtype, intersect):
    # downcasting is enabled, we upcast to the highest
    if "uint" in str(dtype):
        index = casting_modes_dict["uint"]().index(dtype) - 1
        result = ""
        while index >= 0:
            if casting_modes_dict["int"]()[index] not in intersect:
                result = casting_modes_dict["uint"]()[index]
                break
            index -= 1
        return result

    if "int" in dtype:
        index = casting_modes_dict["int"]().index(dtype) - 1
        result = ""
        while index >= 0:
            if casting_modes_dict["int"]()[index] not in intersect:
                result = casting_modes_dict["int"]()[index]
                break
            index -= 1
        return result

    if "float" in dtype:
        index = casting_modes_dict["float"]().index(dtype) - 1

        result = ""
        while index >= 0:
            if casting_modes_dict["float"]()[index] not in intersect:
                result = casting_modes_dict["float"]()[index]
                break
            index -= 1
        return result

    if "complex" in dtype:
        index = casting_modes_dict["complex"]().index(dtype) - 1
        result = ""
        while index >= 0:
            if casting_modes_dict["complex"]()[index] not in intersect:
                result = casting_modes_dict["complex"]()[index]
                break
            index -= 1
        return result


def cross_caster(intersect):
    # check if this is an integer unsupported case
    # intersect is unordered, sorting it makes a list
    # and remaking it a set messes the order
    # so we stick with making both of these
    # sorted lists
    dtype = ""
    valid_float = sorted(ivy.valid_float_dtypes)
    valid_int = sorted(ivy.valid_int_dtypes)
    intersect = sorted(intersect)
    if intersect == valid_int:
        # make dtype equal to default float
        dtype = ivy.default_float_dtype()
    elif intersect == valid_float:
        # make dtype equal to default int
        dtype = ivy.default_int_dtype()

    return str(dtype)


def try_array_function_override(func, overloaded_args, types, args, kwargs):
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __ivy_array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __ivy_array_function__ implementation might reasonably need to
        # check all argument types.
        try:
            result = overloaded_arg.__ivy_array_function__(func, types, args, kwargs)
        except Exception:
            raise ivy.utils.exceptions.IvyNotImplementedException

        if result is not NotImplemented:
            return True, result

    raise TypeError(
        "no implementation found for {} on types that implement "
        "__ivy_array_function__: {}".format(func, list(map(type, overloaded_args)))
    )


def _get_first_array(*args, **kwargs):
    # ToDo: make this more efficient, with function ivy.nested_nth_index_where
    array_fn = ivy.is_array if "array_fn" not in kwargs else kwargs["array_fn"]
    arr = None
    if args:
        arr_idxs = ivy.nested_argwhere(args, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(args, arr_idxs[0])
        else:
            arr_idxs = ivy.nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
            if arr_idxs:
                arr = ivy.index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = ivy.nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(kwargs, arr_idxs[0])
    return arr


def _build_view(original, view, fn, args, kwargs, index=None):
    if ivy.exists(original._base):
        if ivy.backend in ("jax", "tensorflow"):
            warnings.warn(
                "Creating many views will lead to overhead "
                "when performing inplace updates with this backend"
            )
        base = original._base
        view._base = base
        view._manipulation_stack = python_copy.copy(original._manipulation_stack)
    else:
        base = original
        view._base = base
    base._view_refs.append(weakref.ref(view))
    view._manipulation_stack.append((fn, args[1:], kwargs, index))

    # Handle attributes for torch functions without native view functionality
    if ivy.exists(original._torch_base):
        view._torch_base = (
            original
            if ivy.exists(original._torch_manipulation)
            else original._torch_base
        )
    else:
        view._torch_base = base
    if fn in _torch_non_native_view_functions:
        view._torch_manipulation = (original, (fn, args[1:], kwargs))
        view._torch_base._torch_view_refs.append(weakref.ref(view))
    return view


_torch_non_native_view_functions = ("flip", "flipud", "rot90", "fliplr")


def _check_in_nested_sequence(sequence, value=None, _type=None):
    """
    Check `sequence` for either a `value` or a value of type `_type`.

    Helper to recursively check if a N-level nested `sequence` contains
    either a `value` or contains a value of type `_type` and return a
    boolean flag.
    """
    if sequence is value or (isinstance(sequence, _type)):
        # Base case - N = 0
        return True
    elif isinstance(sequence, (tuple, list)):
        if any(isinstance(_val, _type) or _val is value for _val in sequence):
            # N = 1
            return True
        else:
            return any(
                _check_in_nested_sequence(sub_sequence, value, _type)
                for sub_sequence in sequence
                if isinstance(sub_sequence, (tuple, list))
            )


# Array Handling #
# ---------------#


def handle_array_function(fn):
    """
    Wrap a function `fn` to be passed to array_function method.

    Wrap a function to extract the relevant argument types to be passed
    to array_function method.
    """

    @functools.wraps(fn)
    def _handle_array_function(*args, **kwargs):
        overloaded_types = []
        overloaded_args = []

        for arg in args + tuple(kwargs.values()):
            if ivy.exists(arg):
                if not isinstance(arg, ivy.Container) and hasattr(
                    arg, "__ivy_array_function__"
                ):
                    if type(arg) not in overloaded_types:
                        overloaded_types.append(type(arg))
                        if (
                            arg.__ivy_array_function__
                            is not ivy.Array.__ivy_array_function__
                            and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                        ):
                            index = len(overloaded_args)
                            for i, old_arg in enumerate(overloaded_args):
                                if issubclass(type(arg), type(old_arg)):
                                    index = i
                                    break
                            overloaded_args.insert(index, arg)
                elif isinstance(arg, ivy.Container):
                    arg = ivy.Container.cont_flatten_key_chains(arg)
                    indices = ivy.nested_argwhere(
                        arg, lambda x: hasattr(x, "__ivy_array_function__")
                    )
                    for a in indices:
                        if type(getattr(arg, a[0])) not in overloaded_types:
                            overloaded_types.append(type(getattr(arg, a[0])))

                            if getattr(
                                arg, a[0]
                            ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                            ):
                                index = len(overloaded_args)
                                for i, old_arg in enumerate(overloaded_args):
                                    if issubclass(
                                        type(getattr(arg, a[0])), type(old_arg)
                                    ):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)

        success, value = try_array_function_override(
            ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, args, kwargs
        )
        if success:
            return value
        return fn(*args, **kwargs)

    _handle_array_function.handle_array_function = True
    return _handle_array_function


def handle_array_like_without_promotion(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_array_like_without_promotion(*args, **kwargs):
        args = list(args)
        num_args = len(args)
        try:
            type_hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = list(type_hints.keys())
        annotations = [param.annotation for param in type_hints.values()]

        for i, (annotation, parameter, arg) in enumerate(
            zip(annotations, parameters, args)
        ):
            annotation_str = str(annotation)
            if (
                ("rray" in annotation_str or "Tensor" in annotation_str)
                and parameter != "out"
                and all(
                    sq not in annotation_str
                    for sq in ["Sequence", "List", "Tuple", "float", "int", "bool"]
                )
            ):
                if i < num_args:
                    # Fix for ellipsis, slices for numpy's __getitem__
                    # No need to try and convert them into arrays
                    # since asarray throws unpredictable bugs
                    if _check_in_nested_sequence(arg, value=Ellipsis, _type=slice):
                        continue
                    if not ivy.is_array(arg):
                        args[i] = ivy.array(arg)
                elif parameters in kwargs:
                    kwarg = kwargs[parameter]
                    if not ivy.is_array(kwarg):
                        kwargs[parameter] = ivy.array(kwarg)

        return fn(*args, **kwargs)

    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion


def inputs_to_native_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_arrays(*args, **kwargs):
        """
        Convert all `ivy.Array` instances in both the positional and keyword arguments
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
        if not ivy.array_mode:
            return fn(*args, **kwargs)
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.NativeArray instances
        new_args, new_kwargs = ivy.args_to_native(*args, **kwargs)
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

    _inputs_to_native_arrays.inputs_to_native_arrays = True
    return _inputs_to_native_arrays


def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays(*args, **kwargs):
        """
        Convert all `ivy.NativeArray` instances in both the positional and keyword
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
        if not ivy.array_mode:
            warnings.warn(
                "In the case of Compositional function, operators might cause"
                " inconsistent behavior when array_mode is set to False"
            )
            return fn(*args, **kwargs)

        has_out = False
        if "out" in kwargs:
            out = kwargs["out"]
            has_out = True
        # convert all arrays in the inputs to ivy.Array instances
        ivy_args, ivy_kwargs = ivy.args_to_ivy(
            *args, **kwargs, include_derived={tuple: True}
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_arrays.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays


def inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            [args, kwargs],
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
        )
        return fn(*args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


def outputs_to_ivy_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_ivy_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            [args, kwargs],
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
        )
        return fn(*args, **kwargs)

    _outputs_to_ivy_shapes.outputs_to_ivy_shapes = True
    return _outputs_to_ivy_shapes


def to_native_shapes_and_back(fn: Callable) -> Callable:
    """
    Make `fn` receive `ivy.NativeShape` and return `ivy.Shape`.

    Wrap `fn` so that input shapes are all converted to
    `ivy.NativeShape` instances and return shapes are all converted to
    `ivy.Shape` instances.
    """
    return outputs_to_ivy_shapes(inputs_to_native_shapes(fn))


def outputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
        """
        Call the function, and then converts all `ivy.NativeArray` instances in the
        function return into `ivy.Array` instances.

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
        return (
            ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
            if ivy.array_mode
            else ret
        )

    _outputs_to_ivy_arrays.outputs_to_ivy_arrays = True
    return _outputs_to_ivy_arrays


def output_to_native_arrays(fn: Callable) -> Callable:
    """
    Call the function, and then converts all `ivy.Array` instances in the function
    return into `ivy.NativeArray` instances.

    Parameters
    ----------
    args
        The arguments to be passed to the function.

    kwargs
        The keyword arguments to be passed to the function.

    Returns
    -------
        The return of the function, with ivy arrays as native arrays.
    """

    @functools.wraps(fn)
    def _output_to_native_arrays(*args, **kwargs):
        ret = fn(*args, **kwargs)
        return ivy.to_native(ret, nested=True, include_derived={tuple: True})

    _output_to_native_arrays.outputs_to_native_arrays = True
    return _output_to_native_arrays


def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    """
    Make `fn` receive `ivy.Array` and return `ivy.NativeArray`.

    Wrap `fn` so that input arrays are all converted to `ivy.Array`
    instances and return arrays are all converted to `ivy.NativeArray`
    instances.
    """
    return output_to_native_arrays(inputs_to_ivy_arrays(fn))


def to_native_arrays_and_back(fn: Callable) -> Callable:
    """
    Make `fn` receive `ivy.NativeArray` and return `ivy.Array`.

    Wrap `fn` so that input arrays are all converted to
    `ivy.NativeArray` instances and return arrays are all converted to
    `ivy.Array` instances.
    """
    return outputs_to_ivy_arrays(inputs_to_native_arrays(fn))


def frontend_outputs_to_ivy_arrays(fn: Callable) -> Callable:
    """
    Wrap `fn` and convert all frontend arrays in its return to ivy arrays.

    Used in cases when a frontend function receives a callable (frontend
    function) argument. To be able to use that callable in a composition
    of ivy functions, its outputs need to be converted to ivy arrays.
    """

    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
        ret = fn(*args, **kwargs)
        return ivy.nested_map(
            ret,
            lambda x: x.ivy_array if hasattr(x, "ivy_array") else x,
            shallow=False,
        )

    return _outputs_to_ivy_arrays


def handle_view(fn: Callable) -> Callable:
    """
    Wrap `fn` and performs view handling if copy is False.

    Used for functional backends (Jax and TensorFlow). Checks if the
    first arg is a view or original array by checking if the ._base
    attribute is populated. If it's original it adds the returned array
    to its view references, then the returned array adds the operation
    to its manipulation stack and stores the original as its base. If
    the first arg is a view, then the returned array copies its base and
    manipulation stack, appends the new operation to the manipulation
    stack and appends its reference to the base array's view_refs
    attribute.
    """

    @functools.wraps(fn)
    def _handle_view(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if ("copy" in kwargs and kwargs["copy"]) or not ivy.is_ivy_array(args[0]):
            return ret
        original = args[0]
        if isinstance(ret, (list, tuple)):
            for i, view in enumerate(ret):
                ret[i] = _build_view(original, view, fn.__name__, args, kwargs, i)
        else:
            ret = _build_view(original, ret, fn.__name__, args, kwargs, None)
        return ret

    _handle_view.handle_view = True
    return _handle_view


def handle_view_indexing(fn: Callable) -> Callable:
    """
    Wrap `fn` and performs view handling specifically for indexing.

    As with NumPy it returns a copy if advanced indexing is performed.
    Used for functional backends (Jax and TensorFlow). Checks if the
    first arg is a view or original array by checking if the ._base
    attribute is populated. If it's original it adds the returned array
    to its view references, then the returned array adds the operation
    to its manipulation stack and stores the original as its base. If
    the first arg is a view, then the returned array copies its base and
    manipulation stack, appends the new operation to the manipulation
    stack and appends its reference to the base array's view_refs
    attribute.
    """

    @functools.wraps(fn)
    def _handle_view_indexing(*args, **kwargs):
        ret = fn(*args, **kwargs)
        if ("copy" in kwargs and kwargs["copy"]) or not ivy.is_ivy_array(args[0]):
            return ret
        query = kwargs["query"] if "query" in kwargs else args[1]
        query = (query,) if not isinstance(query, tuple) else query
        if [i for i in query if not isinstance(i, (slice, int))]:
            return ret
        original = args[0]
        # ToDo: Remove hard coding of only function with this wrapper
        #  Need general way to convert special method to function found in ivy.__dict__
        ret = _build_view(original, ret, "get_item", args, kwargs)
        return ret

    _handle_view_indexing.handle_view_indexing = True
    return _handle_view_indexing


def _convert_numpy_arrays_to_backend_specific(*args):
    if isinstance(args, np.ndarray):
        np_arr_idxs = ivy.nested_argwhere(args, lambda x: isinstance(x, np.ndarray))
        np_arr_val = ivy.multi_index_nest(args, np_arr_idxs)
        backend_arr_vals = [ivy.array(x).to_native() for x in np_arr_val]
        ivy.set_nest_at_indices(args, np_arr_idxs, backend_arr_vals)
    return args


def handle_numpy_arrays_in_specific_backend(fn: Callable) -> Callable:
    """
    Wrap `fn` and converts all `numpy.ndarray` inputs to `torch.Tensor` instances.

    Used for functional backends (PyTorch). Converts all `numpy.ndarray`
    inputs to `torch.Tensor` instances.
    """

    @functools.wraps(fn)
    def _handle_numpy_array_in_torch(*args, **kwargs):
        args = _convert_numpy_arrays_to_backend_specific(*args)
        ret = fn(*args, **kwargs)
        return ret

    _handle_numpy_array_in_torch.handle_numpy_arrays_in_specific_backend = True
    return _handle_numpy_array_in_torch


# Data Type Handling #
# -------------------#


def infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
        """
        Determine the correct `dtype`, and then calls the function with the `dtype`
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
        ivy.utils.assertions._check_jax_x64_flag(dtype)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    _infer_dtype.infer_dtype = True
    return _infer_dtype


# Device Handling #
# ----------------#


def infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _infer_device(*args, device=None, **kwargs):
        """
        Determine the correct `device`, and then calls the function with the `device`
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

    _infer_device.infer_device = True
    return _infer_device


def handle_device_shifting(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_device_shifting(*args, **kwargs):
        """
        Move all array inputs of the function to `ivy.default_device()`.

        Parameters
        ----------
        args
            The arguments to be passed to the function.
        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function.
        """
        if ivy.soft_device_mode:
            return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        inputs = args + tuple(kwargs.values())
        devices = tuple(ivy.dev(x) for x in inputs if ivy.is_native_array(x))
        unique_devices = set(devices)
        # check if arrays are on the same device
        if len(unique_devices) == 1:
            with ivy.DefaultDevice(next(iter(unique_devices))):
                return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        # raise when arrays are on different devices
        elif len(unique_devices) > 1:
            raise ivy.utils.exceptions.IvyException(
                "Expected all input arrays to be on the same device, "
                f"but found atleast two devices - {devices}, "
                "set `ivy.set_soft_device_mode(True)` to handle this problem."
            )
        return fn(*args, **kwargs)

    _handle_device_shifting.handle_device_shifting = True
    return _handle_device_shifting


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn: Callable) -> Callable:
    handle_out_in_backend = hasattr(fn, "support_native_out")

    @functools.wraps(fn)
    def _handle_out_argument(*args, out=None, **kwargs):
        """
        Call `fn` with the `out` argument handled correctly for performing an inplace
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
        nonlocal handle_out_in_backend
        if out is None:
            return fn(*args, out=out, **kwargs)
        if ivy.gradients._is_variable(out):
            handle_out_in_backend = False
        if handle_out_in_backend:
            # extract underlying native array for out
            native_out = ivy.to_native(out)
            # compute return, with backend inplace update handled by
            # the backend function
            ret = fn(*args, out=native_out, **kwargs)
            if isinstance(ret, (tuple, list)):
                for i in range(len(ret)):
                    ivy.inplace_update(out[i], ret[i])
                    if ivy.backend == "torch":
                        _update_torch_views(out[i])
            else:
                ivy.inplace_update(out, ret)
                if ivy.backend == "torch":
                    _update_torch_views(out)
            return out
        # compute return, and then handle the inplace update explicitly

        ret = fn(*args, **kwargs)
        if not ivy.is_array(ret) and not ivy.is_ivy_container(ret):
            return ivy.nested_multi_map(
                lambda x, _: ivy.inplace_update(
                    x[0], ivy.astype(x[1], ivy.dtype(x[0]))
                ),
                [out, ret],
            )
        return ivy.inplace_update(out, ivy.astype(ret, ivy.dtype(out)))
        # return output matches the dtype of the out array to match numpy and torch

    _handle_out_argument.handle_out_argument = True
    return _handle_out_argument


def _update_torch_views(x, visited_view=None):
    if x._torch_view_refs != []:
        _update_torch_references(x, visited_view)
    if ivy.exists(x._torch_manipulation):
        parent_tensor, fn_args_kwargs = x._torch_manipulation
        fn, args, kwargs = fn_args_kwargs
        kwargs["copy"] = True
        if fn == "rot90":
            kwargs = kwargs.copy()
            kwargs["k"] = -kwargs["k"]
            parent_tensor.data[()] = ivy.__dict__[fn](x, *args, **kwargs).data
        else:
            parent_tensor.data[()] = ivy.__dict__[fn](x, *args, **kwargs).data
    if ivy.exists(x._torch_base):
        _update_torch_views(x._torch_base, visited_view=x)


def _update_torch_references(x, visited_view=None):
    for ref in x._torch_view_refs:
        view = ref()
        if ivy.exists(view) and view is not visited_view:
            parent_tensor, fn_args_kwargs = view._torch_manipulation
            fn, args, kwargs = fn_args_kwargs
            kwargs["copy"] = True
            view.data[()] = ivy.__dict__[fn](parent_tensor, *args, **kwargs).data
            if view._torch_view_refs != []:
                _update_torch_references(view)


# Nestable Handling #
# ------------------#


def handle_nestable(fn: Callable) -> Callable:
    fn_name = fn.__name__

    @functools.wraps(fn)
    def _handle_nestable(*args, **kwargs):
        """
        Call `fn` with the *nestable* property of the function correctly handled. This
        means mapping the function to the container leaves if any containers are passed
        in the input.

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
        if hasattr(ivy.Container, "_static_" + fn_name):
            cont_fn = getattr(ivy.Container, "_static_" + fn_name)
        else:
            cont_fn = lambda *args, **kwargs: ivy.Container.cont_multi_map_in_function(
                fn, *args, **kwargs
            )
        if ivy.nestable_mode and (
            ivy.nested_any(args, ivy.is_ivy_container, check_nests=True)
            or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True)
        ):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    _handle_nestable.handle_nestable = True
    return _handle_nestable


def handle_ragged(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_ragged(*args, **kwargs):
        """
        Call `fn` with the *ragged* property of the function correctly handled. This
        means mapping the function to the RaggedArray arrays if any RaggedArrays are
        passed in the input.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with the ragged property handled correctly.
        """
        nested_fn = (
            lambda *args, **kwargs: ivy.NestedArray.ragged_multi_map_in_function(
                fn, *args, **kwargs
            )
        )
        if ivy.nested_any(
            args, ivy.is_ivy_nested_array, check_nests=True
        ) or ivy.nested_any(kwargs, ivy.is_ivy_nested_array, check_nests=True):
            return nested_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    _handle_ragged.handle_ragged = True
    return _handle_ragged


# Partial Mixed Function Handling #


def handle_partial_mixed_function(fn) -> Callable:
    @functools.wraps(fn)
    def _handle_partial_mixed_function(*args, **kwargs):
        handle_mixed_in_backend = False
        if not hasattr(fn, "partial_mixed_handler"):
            handle_mixed_in_backend = True
        else:
            compos = getattr(fn, "compos")
            condition = getattr(fn, "partial_mixed_handler")

        if handle_mixed_in_backend or condition(*args, **kwargs):
            return fn(*args, **kwargs)
        return compos(*args, **kwargs)

    _handle_partial_mixed_function.handle_partial_mixed_function = True
    return _handle_partial_mixed_function


# Functions #


def _wrap_function(
    key: str, to_wrap: Callable, original: Callable, compositional: bool = False
) -> Callable:
    """
    Apply wrapping to backend implementation `to_wrap` if the original implementation
    `original` is also wrapped, and if `to_wrap` is not already wrapped. Attributes
    `handle_nestable`, `infer_device` etc are set during wrapping, hence indicate to us
    whether a certain function has been wrapped or not. Also handles wrapping of the
    `linalg` namespace.

    Parameters
    ----------
    to_wrap
        the new implementation to potentially wrap
    original
        the original implementation of `to_wrap` which tells us which wrappers we need.
    compositional
        indicates whether the function being wrapped is compositional
        (Default Value = ``False``).

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
                and linalg_k.lower() != "namedtuple"
                and linalg_k != "with_unsupported_dtypes"
                and not linalg_k.startswith("_")
            ):
                to_wrap.__dict__[linalg_k] = _wrap_function(
                    linalg_k,
                    linalg_v,
                    ivy.__dict__[linalg_k],
                    compositional=compositional,
                )
        return to_wrap
    if isinstance(to_wrap, FunctionType):
        # set attributes
        for attr in original.__dict__.keys():
            # private attribute or decorator
            if (
                attr.startswith("_")
                or hasattr(ivy, attr)
                or attr == "mixed_backend_wrappers"
            ):
                continue
            setattr(to_wrap, attr, getattr(original, attr))
        # Copy docstring
        docstring_attr = ["__annotations__", "__doc__"]
        for attr in docstring_attr:
            setattr(to_wrap, attr, getattr(original, attr))

        mixed_fn = hasattr(original, "mixed_backend_wrappers") and original != to_wrap
        partial_mixed = (
            mixed_fn
            and hasattr(original, "handle_partial_mixed_function")
            and hasattr(to_wrap, "partial_mixed_handler")
        )
        add_wrappers, skip_wrappers = [], []
        if mixed_fn:
            backend_wrappers = getattr(original, "mixed_backend_wrappers")
            add_wrappers = backend_wrappers.get("to_add")
            skip_wrappers = backend_wrappers.get("to_skip")

        for attr in FN_DECORATORS:
            if hasattr(original, attr) and not hasattr(to_wrap, attr):
                if partial_mixed and attr == "handle_partial_mixed_function":
                    to_wrap.compos = original
                    to_wrap = handle_partial_mixed_function(to_wrap)
                if attr not in skip_wrappers:
                    to_wrap = getattr(ivy, attr)(to_wrap)
            if attr in add_wrappers:
                to_wrap = getattr(ivy, attr)(to_wrap)

        # we should remove the all the decorators
        # after handle_mixed_fuction in FN_DECORATORS
        # from the compos function because these will
        # be run from the primary implementation.
        if partial_mixed:
            array_spec = to_wrap.compos.__dict__["array_spec"]
            for attr in FN_DECORATORS[
                -1 : FN_DECORATORS.index("handle_partial_mixed_function") : -1
            ]:
                if hasattr(to_wrap.compos, attr):
                    to_wrap.compos = to_wrap.compos.__wrapped__
            to_wrap.compos.__dict__["array_spec"] = array_spec
    return to_wrap


def casting_modes_ops(fn):
    @functools.wraps(fn)
    def method(*args, **kwargs):
        # we first check if it has unsupported/supported dtypes uniquely added to it
        intersect = set(ivy.function_unsupported_dtypes(fn)).difference(
            set(ivy.invalid_dtypes)
        )
        if not intersect:
            # doesn't have unsupported dtypes specified
            # so check if it's one of the device_and_dtype one
            intersect = set(
                ivy.function_unsupported_devices_and_dtypes(fn).get(
                    ivy.default_device().split(":")[0], {None}
                )
            ).difference(set(ivy.invalid_dtypes))
            if not intersect:
                # no unsupported dtype specified
                return fn(*args, **kwargs)

        if "dtype" in kwargs and kwargs["dtype"] is not None:
            dtype = caster(kwargs["dtype"], intersect)
            if dtype:
                kwargs["dtype"] = ivy.as_native_dtype(dtype)

        def mini_helper(x):
            if not hasattr(x, "dtype"):
                return x
            dtype = caster(x, intersect)
            if dtype:
                x = ivy.to_native(ivy.astype(x, ivy.as_native_dtype(dtype)))
            return x

        args = ivy.nested_map(args, mini_helper, include_derived=True)
        kwargs = ivy.nested_map(kwargs, mini_helper)
        return fn(*args, **kwargs)

    return method


# Gets dtype from a version dictionary
def _dtype_from_version(dic, version):
    # if version is a string, it's a frontend function
    if isinstance(version, str):
        version = ivy.functional.frontends.__dict__["versions"][version]
    # if version is a dict, extract the version
    if isinstance(version, dict):
        version = version["version"]

    # If version dict is empty, then there is an error
    if not dic:
        raise Exception("No version found in the dictionary")

    # If key is already in the dictionary, return the value
    if version in dic:
        return dic[version]

    version_tuple = tuple(map(int, version.split(".")))

    # If key is not in the dictionary, check if it's in any range
    # three formats are supported:
    # 1. x.y.z and above
    # 2. x.y.z and below
    # 3. x.y.z to x.y.z
    for key in dic.keys():
        kl = key.split(" ")
        k1 = tuple(map(int, kl[0].split(".")))
        if "above" in key and k1 <= version_tuple:
            return dic[key]
        if "below" in key and k1 >= version_tuple:
            return dic[key]
        if "to" in key and k1 <= version_tuple <= tuple(map(int, kl[2].split("."))):
            return dic[key]

    # if no version is found, we return empty tuple
    return ()


def _versioned_attribute_factory(attribute_function, base):
    class VersionedAttributes(base):
        """
        Class which add versioned attributes to a class, inheriting from `base`.

        Create a class which inherits `base` this way if isinstance is
        called on an instance of the class, it will return True if
        testing for the baseclass, such as isinstance(instance, tuple)
        if `base` is tuple.
        """

        def __init__(self):
            self.attribute_function = attribute_function

        def __get__(self, instance=None, owner=None):
            # version dtypes recalculated everytime it's accessed
            return self.attribute_function()

        def __iter__(self):
            # iter allows for iteration over current version that's selected
            return iter(self.__get__())

        def __repr__(self):
            return repr(self.__get__())

    return VersionedAttributes()


def _dtype_device_wrapper_creator(attrib, t):
    """
    Create a wrapper for a dtype or device attribute.

    The wrapper returns the correct dtype or device for the current version of the
    backend.

    Parameters
    ----------
    attrib
        The attribute name to be wrapped. for example, "unsupported_dtypes"
    t
        The type of the attribute. for example, "tuple"

    Returns
    -------
    A wrapper function for the attribute.
    """

    def _wrapper_outer(version_dict, version, exclusive=True):
        def _wrapped(func):
            val = _versioned_attribute_factory(
                lambda: _dtype_from_version(version_dict, version), t
            )
            if hasattr(func, "override"):
                # we do nothing
                return func
            if not exclusive:
                # exclusive attribute comes into existence
                # only when exlusive is passed as true
                setattr(func, "exclusive", True)
            # set the attribute on the function and return the function as is

            has_attrib = [
                attribute for attribute in attribute_dict if hasattr(func, attribute)
            ] or False
            if has_attrib:
                for attribs in has_attrib:
                    if not (
                        attrib == attribs or (attrib, attribs) in attribute_conflict
                    ):
                        # cases when we encounter two different decorators
                        # applied to the function, but they are not same
                        # and aren't in conflicting dict either
                        setattr(func, attrib, val)
                        setattr(func, "dictionary_info", version_dict)
                    elif hasattr(func, "exclusive"):
                        if attrib == attribs:
                            # we see a higher decorator with exclusivity applied
                            # we use this decorator's dict information
                            # and previous decorator's dict information
                            # to update this
                            old_version_dict = getattr(func, "dictionary_info")
                            old_version_dict.update(version_dict)
                            val = _versioned_attribute_factory(
                                lambda: _dtype_from_version(
                                    version_dict, old_version_dict
                                ),
                                t,
                            )
                            setattr(func, attrib, val)
                        else:
                            # for conflicting ones we do nothing
                            pass
            else:
                setattr(func, attrib, val)
                setattr(func, "dictionary_info", version_dict)
            if "frontends" in func.__module__:
                # it's a frontend func, no casting modes for this
                return func
            return casting_modes_ops(func)

        return _wrapped

    return _wrapper_outer


# nans Handling #
# --------------#


def _leaf_has_nans(x):
    if isinstance(x, ivy.Container):
        return x.has_nans()
    elif ivy.is_array(x):
        return ivy.isnan(x).any()
    elif x is float("nan"):
        return True
    return False


def _nest_has_nans(x):
    return ivy.nested_any(x, _leaf_has_nans)


def handle_nans(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_nans(*args, **kwargs):
        """
        Check for the existence of nans in all arrays in the `args` and `kwargs`.

        The presence of nans is then handled depending on the enabled `nan_policy`.

        Following policies apply:
        raise_exception: raises an exception in case nans are present
        warns: warns a user in case nans are present
        nothing: does nothing

        Parameters
        ----------
        args
            The arguments to be passed to the function.
        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with handling of inputs based
            on the selected `nan_policy`.
        """
        nan_policy = ivy.nan_policy
        # skip the check if the current nan policy is `nothing``
        if nan_policy == "nothing":
            return fn(*args, **kwargs)

        # check all args and kwards for presence of nans
        result = _nest_has_nans(args) or _nest_has_nans(kwargs)

        if result:
            # handle nans based on the selected policy
            if nan_policy == "raise_exception":
                raise ivy.utils.exceptions.IvyException(
                    "Nans are not allowed in `raise_exception` policy."
                )
            elif nan_policy == "warns":
                logging.warning("Nans are present in the input.")

        return fn(*args, **kwargs)

    _handle_nans.handle_nans = True
    return _handle_nans


# Complex number handling #
# ----------------------- #
def handle_complex_input(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_complex_input(
        inp,
        *args,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        **kwargs,
    ):
        """
        Check whether the first positional argument is an array of complex type, and if
        so handle it according to the provided `complex_mode`.

        The options are:
        `"jax"` (default): emulate the behaviour of the JAX framework. If the function
            has a `jax_like` attribute then this will be used to decide on the
            behaviour (see below) and if not, then the entire array will be passed to
            the function.
        `"split"`: execute the function separately on the real and imaginary parts of
            the input.
        `"magnitude"`: execute the function on the magnitude of the input, and keep the
            angle constant.

        The `jax_like` attribute (which should be added to the function itself, and not
        passed as a parameter) has the following options:
        `"entire"` (default): pass the entire input to the function. This is best used
            for purely mathematical operators which are already well defined on complex
            inputs, as many backends will throw exceptions otherwise.
        `"split"`: as the `"split"` option for `complex_mode`
        `"magnitude"`: as the `"magnitude"` option for `complex_mode`
        A callable function: the function will be called instead of the originally
            decorated function. It will be passed `inp` and `*args` as positional
            arguments, and the original `**kwargs` plus `fn_original` as keyword
            arguments. The latter is the original function, in case the `jax_like`
            function wishes to call it.

        Parameters
        ----------
        inp
            The first positional argument to the function, which is expected to be an
            :class:`ivy.Array`.
        args
            The remaining positional arguments to be passed to the function.
        complex_mode
            Optional argument which specifies the method that will be used to handle
            the input, if it is complex.
        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with handling of inputs based
            on the selected `complex_mode`.

        Examples
        --------
        Using the default `jax_like` behaviour
        >>> @handle_complex_input
        >>> def my_func(inp):
        >>>     return ivy.ones_like(inp)

        >>> x = ivy.array([1+1j, 3+4j, 5+12j])
        >>> my_func(x)  # equivalent to setting complex_mode="jax"
        ivy.array([1.+0.j, 1.+0.j, 1.+0.j])

        >>> my_func(x, complex_mode="split")
        ivy.array([1.+1.j, 1.+1.j, 1.+1.j])

        >>> my_func(x, complex_mode="magnitude")
        ivy.array([0.70710681+0.70710675j, 0.60000001+0.79999999j,
                   0.38461535+0.92307694j])

        Using non-default `jax_like` behaviour
        >>> @handle_complex_input
        >>> def my_func(inp):
        >>>     return ivy.ones_like(inp)
        >>> my_func.jax_like = "split"
        >>> my_func(x, complex_mode="jax")
        ivy.array([1.+1.j, 1.+1.j, 1.+1.j])

        Using callable `jax_like` behaviour
        >>> def _my_func_jax_like(inp, fn_original=None):
        >>>     return fn_original(inp) * 3j
        >>> @handle_complex_input
        >>> def my_func(inp):
        >>>     return ivy.ones_like(inp)
        >>> my_func.jax_like = _my_func_jax_like
        >>> my_func(x, complex_mode="jax")
        ivy.array([0.+3.j, 0.+3.j, 0.+3.j])
        """
        if not ivy.is_complex_dtype(inp):
            return fn(inp, *args, **kwargs)

        jax_like = fn.jax_like if hasattr(fn, "jax_like") else "entire"

        if complex_mode == "split" or (complex_mode == "jax" and jax_like == "split"):
            real_inp = ivy.real(inp)
            imag_inp = ivy.imag(inp)
            return fn(real_inp, *args, **kwargs) + 1j * fn(imag_inp, *args, **kwargs)

        elif complex_mode == "magnitude" or (
            complex_mode == "jax" and jax_like == "magnitude"
        ):
            mag_inp = ivy.abs(inp)
            angle_inp = ivy.angle(inp)
            return fn(mag_inp, *args, **kwargs) * ivy.exp(1j * angle_inp)

        elif complex_mode == "jax" and jax_like == "entire":
            return fn(inp, *args, **kwargs)

        elif complex_mode == "jax":
            return jax_like(inp, *args, **kwargs, fn_original=fn)

        else:
            raise IvyValueError(f"complex_mode '{complex_mode}' is not recognised.")

    _handle_complex_input.handle_complex_input = True
    return _handle_complex_input


attribute_dict = {
    "unsupported_dtypes",
    "supported_dtypes",
    "unsupported_devices",
    "supported_devices",
    "unsupported_device_and_dtype",
    "supported_device_and_dtype",
}


attribute_conflict = {
    ("unsupported_devices", "supported_devices"),
    ("supported_devices", "unsupported_devices"),
    ("unsupported_device_and_dtype", "supported_device_and_dtype"),
    ("supported_device_and_dtype", "unsupported_device_and_dtype"),
}

# TODO see if the globals_getter_func can be hacked to return
# the globals in the module where it is working


def globals_getter_func(x=None):
    # define and assign this function to
    # ivy.func_wrapper.globals_getter_func in the module
    # where you want to use the decorators as a context
    # manager
    if not x:
        return globals()
    else:
        globals()[x[0]] = x[1]


class with_unsupported_dtypes(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("unsupported_dtypes", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "unsupported_dtypes", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class with_supported_dtypes(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("supported_dtypes", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "supported_dtypes", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class with_unsupported_devices(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("unsupported_devices", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "unsupported_devices", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class with_supported_devices(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("supported_devices", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "supported_devices", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class with_unsupported_device_and_dtypes(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        # arg inspection
        dicti = args[0]
        self.kwargs = kwargs
        # iterate through the keys
        for key in dicti.keys():
            # maintain a dictionary for nested dictionary
            nested_dic = {}
            for nested_key in dicti[key].keys():
                if nested_key == "all":
                    nested_dic["cpu"] = dicti[key].get("cpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                    nested_dic["tpu"] = dicti[key].get("tpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                    nested_dic["gpu"] = dicti[key].get("gpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                else:
                    nested_dic[nested_key] = dicti[key].get(nested_key, ()) + tuple(
                        dicti[key][nested_key]
                    )
            dicti[key] = nested_dic
        args = (dicti, args[1])

        self.args = args
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("unsupported_device_and_dtype", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals.keys()))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "unsupported_device_and_dtype", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class with_supported_device_and_dtypes(contextlib.ContextDecorator):
    def __init__(self, *args, **kwargs):
        # arg inspection
        dicti = args[0]
        self.kwargs = kwargs
        # iterate through the keys
        for key in dicti.keys():
            # maintain a dictionary for nested dictionary
            nested_dic = {}
            for nested_key in dicti[key].keys():
                if nested_key == "all":
                    nested_dic["cpu"] = dicti[key].get("cpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                    nested_dic["tpu"] = dicti[key].get("tpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                    nested_dic["gpu"] = dicti[key].get("gpu", ()) + tuple(
                        dicti[key]["all"]
                    )
                else:
                    nested_dic[nested_key] = dicti[key].get(nested_key, ()) + tuple(
                        dicti[key][nested_key]
                    )
            dicti[key] = nested_dic
        args = (dicti, args[1])

        self.args = args
        self.globals = {}

    def __call__(self, func=None):
        if func:
            return (
                _dtype_device_wrapper_creator("supported_device_and_dtype", tuple)(
                    *self.args, **self.kwargs
                )
            )(func)

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func(
                        [
                            item,
                            (
                                _dtype_device_wrapper_creator(
                                    "supported_device_and_dtype", tuple
                                )(*self.args, **self.kwargs)
                            )(globals_getter_func()[item]),
                        ]
                    )


class override(contextlib.ContextDecorator):
    def __call__(self, func=None):
        if func:
            setattr(func, "override", "override")
            return func

    def __enter__(self):
        self.globals = globals_getter_func().copy()  # global snapshot

    def __exit__(self, *exec):
        new_globals = set(globals().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    # we need to add the decorator
                    globals_getter_func([item, "override"])
