"""
Function wrapping module for Ivy.

This module provides a comprehensive function wrapping system for Ivy that handles 
type conversions, device management, backend compatibility, and various array 
operations. It contains decorators and helper functions that ensure Ivy functions 
work consistently across different backends (PyTorch, TensorFlow, JAX, NumPy, etc.).

The module includes the following main components:

1. **Type Casting and Promotion**: Functions to handle dtype casting according to 
   different casting modes (upcast, downcast, crosscast) and promotion rules.

2. **Array Conversion Wrappers**: Decorators that convert between ivy.Array and 
   ivy.NativeArray instances, ensuring proper data flow between Ivy's unified 
   interface and backend-specific array types.

3. **Device Handling**: Functions to manage array placement on different devices 
   (CPU, GPU, TPU) and handle device consistency across operations.

4. **Backend Validation**: Utilities to ensure array operations are performed 
   with the correct backend and handle backend-specific quirks.

5. **Complex Number Support**: Specialized handling for complex-valued arrays 
   across different backends with varying complex number support.

6. **View and Memory Management**: Functions to handle array views, memory 
   sharing, and manipulation stack tracking for functional backends.

7. **Context Managers**: Classes for temporarily restricting or enabling 
   specific dtypes and devices during function execution.

The wrapper system follows a specific order defined in FN_DECORATORS to ensure 
proper layering of functionality. This allows Ivy to provide a unified interface 
while maintaining backend-specific optimizations and handling edge cases.

Key Features:
- Automatic type promotion and casting
- Device consistency enforcement
- Memory-efficient view handling
- NaN and exception handling
- Nestable container support
- Frontend/backend array conversion
- Version-specific attribute management

This module is central to Ivy's ability to provide framework-agnostic array 
programming while maintaining compatibility with backend-specific features.
"""

import contextlib
import copy as python_copy
import functools
import inspect
import logging
import numpy as np
from types import FunctionType
from typing import Callable, Literal
import warnings
import weakref

import ivy
from ivy.utils.exceptions import IvyValueError


# for wrapping (sequence matters)
FN_DECORATORS = [
    "handle_complex_input",
    "handle_device",
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
    "handle_backend_invalid",
    "temp_asarray_wrapper",
    "handle_exceptions",
    "handle_nans",
]


# Helpers #
# --------#

# for casting modes, order is the hierarchy
casting_modes_dict = {
    "uint": lambda: ivy.valid_uint_dtypes,
    "int": lambda: sorted(
        set(ivy.valid_int_dtypes).difference(set(ivy.valid_uint_dtypes))
    ),
    "float": lambda: ivy.valid_float_dtypes,
    "complex": lambda: ivy.valid_complex_dtypes,
}


def caster(dtype, intersect):
    """
    Determine the appropriate dtype to cast to based on casting modes.

    This function checks if a given dtype needs to be cast to a different 
    supported dtype based on the current casting mode settings and a set 
    of unsupported dtypes (intersect).

    Parameters
    ----------
    dtype : ivy.Dtype or array-like
        The input dtype to potentially cast. Can be a dtype object or an 
        array with a dtype attribute.
    intersect : set
        Set of unsupported dtypes that should be avoided.

    Returns
    -------
    str or None
        The target dtype to cast to, or None if no casting is needed.

    Notes
    -----
    The function respects the following casting mode hierarchy:
    - cast_dtypes(): Enables all casting types
    - crosscast_dtypes: Enables cross-type casting
    - upcast_dtypes: Enables upcasting to higher precision
    - downcast_dtypes: Enables downcasting to lower precision
    """
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


def cast_helper(arg, dtype, intersect, is_upcast=True):
    """
    Helper function to find the next available dtype in a casting hierarchy.

    This function traverses the dtype hierarchy (uint, int, float, complex) 
    to find the next supported dtype in the specified direction (up or down).

    Parameters
    ----------
    arg : str
        The dtype category ("uint", "int", "float", or "complex").
    dtype : ivy.Dtype
        The current dtype to cast from.
    intersect : set
        Set of unsupported dtypes to avoid.
    is_upcast : bool, optional
        Whether to cast up (True) or down (False) the hierarchy. Default is True.

    Returns
    -------
    str
        The next available dtype in the hierarchy, or empty string if none found.

    Notes
    -----
    Upcasting moves to higher precision (e.g., int8 -> int16 -> int32).
    Downcasting moves to lower precision (e.g., float64 -> float32 -> float16).
    """
    step = 1 if is_upcast else -1
    index = casting_modes_dict[arg]().index(dtype) + step
    result = ""
    while 0 <= index < len(casting_modes_dict[arg]()):
        if casting_modes_dict[arg]()[index] not in intersect:
            result = casting_modes_dict[arg]()[index]
            break
        index += step

    return result


def upcaster(dtype, intersect):
    """
    Cast a dtype to a higher precision type within the same category.

    This function attempts to upcast a dtype to the next higher precision 
    dtype within the same category (uint, int, float, complex) that is 
    not in the unsupported set.

    Parameters
    ----------
    dtype : ivy.Dtype
        The dtype to upcast from.
    intersect : set
        Set of unsupported dtypes to avoid.

    Returns
    -------
    str
        The higher precision dtype to cast to, or empty string if no suitable 
        dtype is found.

    Examples
    --------
    >>> upcaster("int8", {"int16"})
    "int32"

    >>> upcaster("float32", {"float64"})
    ""
    """
    # upcasting is enabled, we upcast to the highest
    if "uint" in str(dtype):
        return cast_helper("uint", dtype, intersect, is_upcast=True)
    if "int" in dtype:
        return cast_helper("int", dtype, intersect, is_upcast=True)
    if "float" in dtype:
        return cast_helper("float", dtype, intersect, is_upcast=True)
    if "complex" in dtype:
        return cast_helper("complex", dtype, intersect, is_upcast=True)


def downcaster(dtype, intersect):
    """
    Cast a dtype to a lower precision type within the same category.

    This function attempts to downcast a dtype to the next lower precision 
    dtype within the same category (uint, int, float, complex) that is 
    not in the unsupported set.

    Parameters
    ----------
    dtype : ivy.Dtype
        The dtype to downcast from.
    intersect : set
        Set of unsupported dtypes to avoid.

    Returns
    -------
    str
        The lower precision dtype to cast to, or empty string if no suitable 
        dtype is found.

    Examples
    --------
    >>> downcaster("int32", {"int16"})
    "int8"
    """
    # downcasting is enabled, we upcast to the highest
    if "uint" in str(dtype):
        return cast_helper("uint", dtype, intersect, is_upcast=False)
    if "int" in dtype:
        return cast_helper("int", dtype, intersect, is_upcast=False)
    if "float" in dtype:
        return cast_helper("float", dtype, intersect, is_upcast=False)
    if "complex" in dtype:
        return cast_helper("complex", dtype, intersect, is_upcast=False)


def cross_caster(intersect):
    """
    Perform cross-category dtype casting based on supported dtypes.

    This function implements cross-category casting by checking if entire 
    dtype categories are unsupported and providing alternative default dtypes.
    For example, if all integer types are unsupported, it may return a 
    default float dtype.

    Parameters
    ----------
    intersect : set
        Set of unsupported dtypes.

    Returns
    -------
    str
        A default dtype for cross-casting, or empty string if no 
        cross-casting is applicable.

    Notes
    -----
    Cross-casting rules:
    - If all int dtypes are unsupported -> use default float dtype
    - If all float/bool dtypes are unsupported -> use default int dtype
    """
    # check if this is an integer unsupported case
    # intersect is unordered, sorting it makes a list
    # and remaking it a set messes the order
    # so we stick with making both of these
    # sorted lists
    dtype = ""
    valid_float = sorted(ivy.valid_float_dtypes)
    valid_int = sorted(ivy.valid_int_dtypes)
    valid_bool = [ivy.bool]
    intersect = sorted(intersect)
    if set(valid_int).issubset(intersect):
        # make dtype equal to default float
        dtype = ivy.default_float_dtype()
    elif set(valid_float).issubset(intersect) or set(valid_bool).issubset(intersect):
        # make dtype equal to default int
        dtype = ivy.default_int_dtype()

    return str(dtype)


def try_array_function_override(func, overloaded_args, types, args, kwargs):
    """
    Attempt to call the __ivy_array_function__ override for the given function.

    This function checks if any of the provided arguments implement the
    __ivy_array_function__ protocol, and if so, attempts to call the override
    with the given function, types, arguments, and keyword arguments.

    Parameters
    ----------
    func : callable
        The function to be overridden.
    overloaded_args : list
        List of arguments that may implement __ivy_array_function__.
    types : tuple
        Tuple of argument types.
    args : tuple
        Positional arguments to pass to the override.
    kwargs : dict
        Keyword arguments to pass to the override.

    Returns
    -------
    tuple
        (True, result) if an override was found and returned a result other than NotImplemented,
        (False, None) if no override was found.
    """
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __ivy_array_function__ on the *first*
        # occurrence of each argument type. This is necessary for reasonable
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
        f"no implementation found for {func} on types that implement"
        f" __ivy_array_function__: {list(map(type, overloaded_args))}"
    )


def _get_first_array(*args, **kwargs):
    """
    Find and return the first array in the provided arguments.

    This function searches through positional and keyword arguments to find 
    the first array-like object, which can be used for device and dtype 
    inference purposes.

    Parameters
    ----------
    *args
        Positional arguments to search through.
    **kwargs
        Keyword arguments to search through. Can include 'array_fn' to 
        customize the array detection function.

    Returns
    -------
    array or None
        The first array found in the arguments, or None if no array is found.

    Notes
    -----
    The function uses ivy.nested_argwhere to recursively search nested 
    structures like lists and tuples. It checks for both ivy.Array and 
    objects with an '_ivy_array' attribute.
    """
    # ToDo: make this more efficient, with function ivy.nested_nth_index_where
    def array_fn(x):
        return (
            ivy.is_array(x)
            if not hasattr(x, "_ivy_array")
            else ivy.is_array(x.ivy_array)
        )

    array_fn = array_fn if "array_fn" not in kwargs else kwargs["array_fn"]
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
    """
    Build a view relationship between original and view arrays.

    This function establishes a view relationship for functional backends
    (JAX and TensorFlow) by setting up the base reference, manipulation
    stack, and view references. It handles both regular views and
    PyTorch-specific non-native view functions.

    Parameters
    ----------
    original : ivy.Array
        The original array that the view is created from.
    view : ivy.Array
        The view array to be configured.
    fn : str
        The name of the function that created the view.
    args : tuple
        The arguments passed to the view-creating function (excluding the first).
    kwargs : dict
        The keyword arguments passed to the view-creating function.
    index : int, optional
        The index for multi-output functions. Default is None.

    Returns
    -------
    ivy.Array
        The configured view array with proper base references and manipulation
        stack.

    Notes
    -----
    This function handles the complex relationships needed for view tracking
    in functional backends where true views don't exist natively.
    """
    if ivy.exists(original._base):
        base = original._base
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


def _get_preferred_device(args, kwargs):
    """
    Determine the preferred device for array creation.

    When new arrays are created, they should be created on the same device as
    existing array inputs. If a device is specified as a kwarg, create them there.
    If not, scan for any other inputs which are already arrays and use the device
    of the first one found (unless we're in soft device mode).

    Parameters
    ----------
    args : tuple
        Positional arguments that may contain arrays.
    kwargs : dict
        Keyword arguments that may contain arrays or a 'device' specification.

    Returns
    -------
    device
        The preferred device for array creation. Returns the specified device
        from kwargs if available, otherwise the device of the first array found,
        or the default device.

    Notes
    -----
    In soft device mode, always returns the default device regardless of
    input array devices.
    """
    device = None
    if "device" in kwargs and kwargs["device"] is not None:
        return device
    if not ivy.soft_device_mode:
        arr_arg = _get_first_array(*args, **kwargs)
        return ivy.default_device(item=arr_arg, as_native=True)
    return ivy.default_device(as_native=True)


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
    """
    Decorator to convert array-like inputs to ``ivy.Array`` without dtype promotion.

    Many backend functions expect true array objects.  This decorator scans the
    positional and keyword arguments *before* calling *fn* and converts any
    plain Python scalars, lists, tuples, or other array-like objects to
    ``ivy.Array`` **without** invoking Ivy's dtype-promotion semantics.  This
    is different from the standard ``inputs_to_ivy_arrays`` wrapper which *may*
    trigger type promotion.  It is therefore useful for low-level wrappers
    where the original dtype must be preserved exactly as provided by the user.

    Parameters
    ----------
    fn : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function with array-like to ``ivy.Array`` conversion.
    """
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

        device = _get_preferred_device(args, kwargs)

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
                    if arg is None or _check_in_nested_sequence(
                        arg, value=Ellipsis, _type=slice
                    ):
                        continue
                    if not ivy.is_array(arg):
                        args[i] = ivy.array(arg, device=device)
                elif parameters in kwargs:
                    kwarg = kwargs[parameter]
                    if not ivy.is_array(kwarg):
                        kwargs[parameter] = ivy.array(kwarg, device=device)

        return fn(*args, **kwargs)

    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion


def inputs_to_native_arrays(fn: Callable) -> Callable:
    """
    Decorator to convert ``ivy.Array`` inputs to backend native arrays.

    Prior to executing *fn*, this decorator traverses all positional and keyword
    arguments and converts every ``ivy.Array`` instance to its underlying
    backend-specific **native** representation (e.g., ``torch.Tensor`` when the
    backend is PyTorch).  The conversion is **deep** – nested containers and
    sequences are handled as well – ensuring the wrapped backend function never
    encounters an ``ivy.Array``.

    Notes
    -----
    * The optional ``out`` argument, if present, is removed **before** the
      conversion and re-inserted afterwards so that an ``ivy.Array`` can still
      be used for inplace updates.
    * The decorator only affects the *inputs*.  Use
      ``outputs_to_ivy_arrays`` if you also need to convert the *outputs* back
      to ``ivy.Array``.
    """
    @functools.wraps(fn)
    def _inputs_to_native_arrays(*args, **kwargs):
        """Convert all `ivy.Array` instances in both the positional and keyword
        arguments into `ivy.NativeArray` instances, and then calls the function
        with the updated arguments.

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
    """
    Decorator to convert backend native array inputs to ``ivy.Array``.

    This is the converse of :func:`inputs_to_native_arrays`.  It ensures that a
    compositional Ivy function always operates on ``ivy.Array`` objects even if
    the caller supplied native backend tensors.  This is crucial because
    compositional implementations often rely on ``ivy`` operators which expect
    Ivy arrays and not backend natives.
    """
    @functools.wraps(fn)
    def _inputs_to_ivy_arrays(*args, **kwargs):
        """Convert all `ivy.NativeArray` instances in both the positional and
        keyword arguments into `ivy.Array` instances, and then calls the
        function with the updated arguments.

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
            *args, **kwargs, include_derived={"tuple": True}
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_arrays.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays


def inputs_to_native_shapes(fn: Callable) -> Callable:
    """
    Decorator to convert ivy.Shape instances to native shapes in function inputs.

    This decorator converts all ivy.Shape instances in the function arguments
    to their underlying native shape representations when array_mode is enabled.

    Parameters
    ----------
    fn : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function that handles shape conversions in inputs.

    Notes
    -----
    This conversion is only applied when ivy.array_mode is True. When False,
    ivy.Shape instances are left unchanged.
    """

    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
            [args, kwargs],
        )
        return fn(*args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


def outputs_to_ivy_shapes(fn: Callable) -> Callable:
    """
    Decorator to convert native shapes to ivy.Shape instances in function outputs.

    This decorator converts native shape representations back to ivy.Shape
    instances in the function's return values when array_mode is enabled.

    Parameters
    ----------
    fn : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function that handles shape conversions in outputs.

    Notes
    -----
    This conversion is only applied when ivy.array_mode is True. When False,
    native shapes are left unchanged.
    """

    @functools.wraps(fn)
    def _outputs_to_ivy_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
            [args, kwargs],
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
    """
    Decorator to convert backend native arrays **in the return value** to ``ivy.Array``.

    After *fn* returns, this decorator walks the returned structure (which can
    be arbitrarily nested) and converts every backend native array to
    ``ivy.Array`` so that downstream Ivy code continues to be framework-agnostic.
    """
    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
        """Call the function, and then converts all `ivy.NativeArray` instances
        in the function return into `ivy.Array` instances.

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
            ivy.to_ivy(ret, nested=True, include_derived={"tuple": True})
            if ivy.array_mode
            else ret
        )

    _outputs_to_ivy_arrays.outputs_to_ivy_arrays = True
    return _outputs_to_ivy_arrays


def output_to_native_arrays(fn: Callable) -> Callable:
    """
    Call the function, and then converts all `ivy.Array` instances in the
    function return into `ivy.NativeArray` instances.

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
        return ivy.to_native(ret, nested=True, include_derived={"tuple": True})

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
            lambda x: x.ivy_array if hasattr(x, "ivy_array") else x,
            ret,
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
        query = query if isinstance(query, tuple) else (query,)
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
    Wrap `fn` and converts all `numpy.ndarray` inputs to `torch.Tensor`
    instances.

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
        """Determine the correct `dtype`, and then calls the function with the
        `dtype` passed explicitly.

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


def handle_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_device(*args, **kwargs):
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
        dev = None
        if "device" in kwargs and kwargs["device"] is not None:
            dev = ivy.as_native_dev(kwargs["device"])
        if ivy.soft_device_mode:
            with ivy.DefaultDevice(ivy.default_device(dev)):
                return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        inputs = args + tuple(kwargs.values())
        devices = tuple(ivy.dev(x) for x in inputs if ivy.is_array(x))
        unique_devices = set(devices)
        # check if arrays are on the same device
        if len(unique_devices) <= 1:
            # len(unique_devices) == 0 when there are no arrays
            dst_dev = (
                dev
                if dev is not None
                else None if len(unique_devices) == 0 else next(iter(unique_devices))
            )
            with ivy.DefaultDevice(ivy.default_device(dst_dev)):
                return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        # raise when arrays are on different devices
        elif len(unique_devices) > 1:
            raise ivy.utils.exceptions.IvyException(
                "Expected all input arrays to be on the same device, "
                f"but found at least two devices - {devices}, "
                "set `ivy.set_soft_device_mode(True)` to handle this problem."
            )
        return fn(*args, **kwargs)

    _handle_device.handle_device = True
    return _handle_device


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn: Callable) -> Callable:
    """
    Decorator implementing NumPy-style ``out`` semantics for Ivy functions.

    Ivy functions support an optional ``out`` keyword for inplace updates.  This
    decorator standardises the behaviour across backends:

    1.  If *out* is ``None`` the function is called normally.
    2.  If the backend supports native ``out`` tensors, delegate the inplace
        update to the backend implementation.
    3.  Otherwise, call the function, then manually copy the result into
        *out* using :func:`ivy.inplace_update`.

    The decorator also handles edge cases such as gradient-tracking variables
    and tuples/lists of outputs.
    """

    handle_out_in_backend = hasattr(fn, "support_native_out")

    @functools.wraps(fn)
    def _handle_out_argument(*args, out=None, **kwargs):
        """
        Call `fn` with the `out` argument handled correctly for performing
        an inplace update.

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
    """
    Update PyTorch views recursively when the base array changes.

    This function propagates changes from a base array to all its views
    for PyTorch backend, handling both direct view references and torch
    manipulation operations.

    Parameters
    ----------
    x : ivy.Array
        The array whose views need to be updated.
    visited_view : ivy.Array, optional
        A view that has already been visited to avoid infinite recursion.
        Default is None.

    Notes
    -----
    This function is specific to PyTorch backend and handles the complex
    view update semantics required for maintaining consistency between
    base arrays and their views.
    """
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
    if ivy.exists(x._torch_base):
        _update_torch_views(x._torch_base, visited_view=x)


def _update_torch_references(x, visited_view=None):
    """
    Update PyTorch view references when the base array changes.

    This function updates all view references of an array by recomputing
    their data based on the current state of their parent tensors.

    Parameters
    ----------
    x : ivy.Array
        The array whose view references need to be updated.
    visited_view : ivy.Array, optional
        A view that has already been visited to avoid infinite recursion.
        Default is None.

    Notes
    -----
    This function works in conjunction with _update_torch_views to maintain
    consistency in the PyTorch view system.
    """
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
        Call `fn` with the *nestable* property of the function correctly
        handled. This means mapping the function to the container leaves if any
        containers are passed in the input.

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
        if hasattr(ivy.Container, f"_static_{fn_name}"):
            cont_fn = getattr(ivy.Container, f"_static_{fn_name}")
        else:

            def cont_fn(*args, **kwargs):
                return ivy.Container.cont_multi_map_in_function(fn, *args, **kwargs)

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
        Call `fn` with the *ragged* property of the function correctly
        handled. This means mapping the function to the RaggedArray arrays if
        any RaggedArrays are passed in the input.

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

        def nested_fn(*args, **kwargs):
            return ivy.NestedArray.ragged_multi_map_in_function(fn, *args, **kwargs)

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
    """
    Decorator to handle partial mixed function implementations.

    This decorator manages functions that have mixed implementations, where
    some backends have native implementations while others use compositional
    fallbacks. It determines whether to use the backend-specific implementation
    or the compositional version based on the function's condition.

    Parameters
    ----------
    fn : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function that routes between native and compositional
        implementations.

    Notes
    -----
    If the function has a 'partial_mixed_handler' attribute, it uses that
    condition to determine which implementation to use. Otherwise, it
    defaults to the backend implementation.
    """
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


# Temporary asarray wrapper (Please request my review before removing)


def temp_asarray_wrapper(fn: Callable) -> Callable:
    """
    Temporary decorator to convert frontend Tensor objects to ivy arrays.

    This decorator converts frontend framework tensors (e.g., torch.Tensor)
    that have an 'ivy_array' attribute to their underlying ivy.Array
    representation before passing them to the function.

    Parameters
    ----------
    fn : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function that handles frontend tensor conversion.

    Notes
    -----
    This is a temporary wrapper and should be used with caution. It specifically
    handles objects with an 'ivy_array' attribute, extracting the underlying
    ivy array for processing.

    The wrapper uses nested_map to recursively convert all frontend tensors
    in the input arguments and keyword arguments.
    """

    @functools.wraps(fn)
    def _temp_asarray_wrapper(*args, **kwargs):
        """
        Convert `Tensor` into `ivy.Array` instances.

        Convert all `Tensor` instances in both the positional and keyword arguments
        into `ivy.Array` instances, and then call the function with the updated
        arguments.
        """

        def _to_ivy_array(x):
            # if x is a frontend torch Tensor (or any frontend "Tensor" actually) return the wrapped ivy array # noqa: E501
            if hasattr(x, "ivy_array"):
                return x.ivy_array
            # else just return x
            return x

        # convert all input arrays to ivy.Array instances
        new_args = ivy.nested_map(
            _to_ivy_array, args, include_derived={"tuple": True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            _to_ivy_array, kwargs, include_derived={"tuple": True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    _temp_asarray_wrapper.temp_asarray_wrapper = True
    return _temp_asarray_wrapper


# Functions #

def _wrap_function(
    key: str, to_wrap: Callable, original: Callable, compositional: bool = False
) -> Callable:
    """
    Apply wrapping to backend implementation `to_wrap` if the original
    implementation `original` is also wrapped, and if `to_wrap` is not already
    wrapped. Attributes `handle_nestable` etc are set during wrapping, hence
    indicate to us whether a certain function has been wrapped or not. Also
    handles wrapping of the `linalg` namespace.

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


def casting_modes_ops(fn, ret_dtype_target=None):
    """
    Apply casting mode operations to a function.

    This decorator wrapper applies dtype casting based on unsupported dtypes
    and the current casting mode settings. It handles argument casting and
    return type casting according to the function's dtype restrictions.

    Parameters
    ----------
    fn : Callable
        The function to apply casting operations to.
    ret_dtype_target : list, optional
        List of argument names to use for return dtype promotion.
        Default is None.

    Returns
    -------
    Callable
        A method that applies casting operations before calling the original
        function and casts the return value if needed.

    Notes
    -----
    The function checks for unsupported dtypes in the function's metadata
    and applies appropriate casting based on the current casting mode
    (upcast, downcast, crosscast). It also handles return type promotion
    when ret_dtype_target is specified.
    """
    @functools.wraps(fn)
    def method(*args, **kwargs):
        # Get the function signature
        signature = inspect.signature(fn)
        # Extract argument names
        arg_names = [param.name for param in signature.parameters.values()]
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

        # specifies which dtype to cast the output to
        to_cast = None
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            to_cast = kwargs["dtype"]
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

        args = ivy.nested_map(mini_helper, args, include_derived=True)
        kwargs = ivy.nested_map(mini_helper, kwargs)

        if not to_cast and ret_dtype_target:
            for arg in ret_dtype_target:
                if arg:
                    to_cast, arg_mod = ivy.promote_types_of_inputs(
                        to_cast,
                        (
                            args[arg_names.index(arg)]
                            if arg not in kwargs
                            else kwargs[arg]
                        ),
                    )
                    if arg not in kwargs:
                        args[arg_names.index(arg)] = (
                            arg_mod
                            if not ivy.is_array(args[arg_names.index(arg)])
                            else args[arg_names.index(arg)]
                        )
                    else:
                        kwargs[arg] = (
                            arg_mod
                            if not ivy.is_array(args[arg_names.index(arg)])
                            else kwargs[arg]
                        )

        return (
            ivy.astype(fn(*args, **kwargs), ivy.to_native(to_cast))
            if to_cast
            else fn(*args, **kwargs)
        )

    return method


# Gets dtype from a version dictionary
def _dtype_from_version(dic, version):
    """
    Extract dtype information from a version-specific dictionary.

    This function retrieves dtype restrictions or attributes based on the
    specified version. It supports various version formats including exact
    matches, ranges, and version bounds.

    Parameters
    ----------
    dic : dict
        Dictionary mapping version strings to dtype information.
    version : str or dict
        Version specification. Can be a version string, a dictionary with
        version information, or a frontend version reference.

    Returns
    -------
    any
        The dtype information corresponding to the specified version.

    Raises
    ------
    ValueError
        If the version dictionary is empty.

    Notes
    -----
    Supported version formats:
    - Exact version: "1.2.3"
    - Version ranges: "1.2.3 to 1.4.0"
    - Lower bounds: "1.2.3 and above"
    - Upper bounds: "1.2.3 and below"

    If no exact match is found, the function returns the last version's
    information as a fallback.
    """
    # if version is a string, it's a frontend function
    if isinstance(version, str):
        version = ivy.functional.frontends.__dict__["versions"][version]
    # if version is a dict, extract the version
    if isinstance(version, dict):
        version = version["version"]

    # If version dict is empty, then there is an error
    if not dic:
        raise ValueError("No version found in the dictionary")

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

    # if no version is found, return the last version
    return dic[list(dic.keys())[-1]]


def _versioned_attribute_factory(attribute_function, base):
    """
    Create a versioned attribute class that inherits from a base type.

    This factory function creates a descriptor class that provides version-aware
    attribute access. The resulting class inherits from the specified base type
    to maintain isinstance compatibility.

    Parameters
    ----------
    attribute_function : Callable
        Function that returns the current version's attribute value.
    base : type
        Base type to inherit from (e.g., tuple, list).

    Returns
    -------
    VersionedAttributes
        A descriptor class that provides version-aware attribute access.

    Notes
    -----
    The created class maintains isinstance compatibility with the base type
    while providing dynamic attribute resolution based on the current
    backend version. This is useful for dtype and device restrictions
    that vary by backend version.
    """
    class VersionedAttributes(base):
        """Class which add versioned attributes to a class, inheriting from
        `base`.

        Create a class which inherits `base` this way if isinstance is
        called on an instance of the class, it will return True if
        testing for the baseclass, such as isinstance(instance, tuple)
        if `base` is tuple.
        """

        def __init__(self):
            self.attribute_function = attribute_function

        def __get__(self, instance=None, owner=None):
            # version dtypes recalculated every time it's accessed
            return self.attribute_function()

        def __iter__(self):
            # iter allows for iteration over current version that's selected
            return iter(self.__get__())

        def __repr__(self):
            return repr(self.__get__())

        def __bool__(self):
            return bool(self.__get__())

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

    def _wrapper_outer(version_dict, version, exclusive=True, ret_dtype_target=None):
        def _wrapped(func):
            val = _versioned_attribute_factory(
                lambda: _dtype_from_version(version_dict, version), t
            )
            if hasattr(func, "override"):
                # we do nothing
                return func
            if not exclusive:
                # exclusive attribute comes into existence
                # only when exclusive is passed as true
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
                        setattr(func, "dictionary_info", (version_dict, version))
                    elif hasattr(func, "exclusive"):
                        if attrib == attribs:
                            # we see a higher decorator with exclusivity applied
                            # we use this decorator's dict information
                            # and previous decorator's dict information
                            # to update this
                            old_version_dict = getattr(func, "dictionary_info")[0]
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
                if not val and attrib.startswith("supported"):
                    setattr(func, f"un{attrib}", val)
                else:
                    setattr(func, attrib, val)
                setattr(func, "dictionary_info", (version_dict, version))
            if "frontends" in func.__module__:
                # it's a frontend func, no casting modes for this
                return func

            return casting_modes_ops(func, ret_dtype_target=ret_dtype_target)

        return _wrapped

    return _wrapper_outer


# nans Handling #
# --------------#


def _leaf_has_nans(x):
    """
    Check if a single value or array contains NaN values.

    This function checks whether a leaf value (single array or scalar) 
    contains any NaN (Not a Number) values.

    Parameters
    ----------
    x : any
        The value to check for NaN values. Can be an ivy.Array,
        ivy.Container, numpy array, or scalar.

    Returns
    -------
    bool
        True if the value contains any NaN values, False otherwise.

    Notes
    -----
    For ivy.Container objects, this delegates to the container's has_nans method.
    For arrays, it uses ivy.isnan to detect NaN values.
    For scalars, it uses numpy.isnan.
    """
    if isinstance(x, ivy.Container):
        return x.has_nans()
    elif ivy.is_array(x):
        return ivy.isnan(x).any()
    elif np.isnan(x):
        return True
    return False


def _nest_has_nans(x):
    """
    Check if a nested structure contains any NaN values.

    This function recursively checks nested structures (lists, tuples, dicts)
    to determine if any leaf values contain NaN values.

    Parameters
    ----------
    x : any
        The nested structure to check for NaN values.

    Returns
    -------
    bool
        True if any value in the nested structure contains NaN, False otherwise.

    Notes
    -----
    This function uses ivy.nested_any to recursively traverse the nested
    structure and apply the _leaf_has_nans function to each leaf.
    """
    return ivy.nested_any(x, _leaf_has_nans)


def handle_nans(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_nans(*args, **kwargs):
        """
        Check for the existence of nans in all arrays in the `args` and
        `kwargs`.

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

        # check all args and kwargs for presence of nans
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
        Check whether the first positional argument is an array of complex
        type, and if so handle it according to the provided `complex_mode`.

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
            real_inp = ivy.real(inp).data
            imag_inp = ivy.imag(inp).data
            if "out" in kwargs and kwargs["out"] is not None:
                out = kwargs.pop("out")
                real_ret = fn(real_inp, *args, out=ivy.real(out), **kwargs)
                imag_ret = fn(imag_inp, *args, out=ivy.imag(out), **kwargs)
            else:
                real_ret = fn(real_inp, *args, **kwargs)
                imag_ret = fn(imag_inp, *args, **kwargs)
            return ivy.add(
                real_ret,
                ivy.multiply(ivy.array(1j, dtype=inp.dtype), imag_ret),
            )

        elif complex_mode == "magnitude" or (
            complex_mode == "jax" and jax_like == "magnitude"
        ):
            mag_inp = ivy.abs(inp).data
            angle_inp = ivy.angle(inp).data
            return ivy.multiply(
                fn(mag_inp, *args, **kwargs), ivy.exp(ivy.multiply(1j, angle_inp))
            )

        elif complex_mode == "jax" and jax_like == "entire":
            return fn(inp, *args, **kwargs)

        elif complex_mode == "jax":
            return jax_like(inp, *args, **kwargs, fn_original=fn)

        else:
            raise IvyValueError(f"complex_mode '{complex_mode}' is not recognised.")

    _handle_complex_input.handle_complex_input = True
    return _handle_complex_input


def handle_backend_invalid(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_backend_invalid(*args, **kwargs):
        """Check if any of the arguments (or nested arguments) passed to the
        function are instances of ivy.Array or ivy.NativeArray. If so, it
        returns the function. If not, it raises an InvalidBackendException.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function if the current
            backend matches the argument backend.
            If not, it raises an InvalidBackendException
        """
        array_indices = ivy.nested_argwhere(
            [args, kwargs], lambda x: isinstance(x, ivy.Array)
        )
        array_vals = ivy.multi_index_nest([args, kwargs], array_indices)

        def func(x):
            target_backend = ivy.utils.backend.handler._determine_backend_from_args(x)
            if (
                target_backend is not None
                and ivy.backend != ""
                and ivy.current_backend_str() != target_backend.backend
            ):
                raise ivy.utils.exceptions.IvyInvalidBackendException(
                    "Operation not allowed. Array was instantiated with backend"
                    f" {target_backend.backend}. But current backend is"
                    f" {ivy.backend}. Please set dynamic=True"
                    " for the array if you want to convert it to the target"
                    " backend"
                )
            return x

        ivy.nested_map(func, array_vals, include_derived=True)

        return fn(*args, **kwargs)

    _handle_backend_invalid.handle_backend_invalid = True
    return _handle_backend_invalid


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
    """
    Get or set global variables in the current module context.

    This function provides access to the global namespace of the module
    where it's called. It can either return the globals dictionary or
    set a specific global variable.

    Parameters
    ----------
    x : list or None, optional
        If None, returns the globals dictionary.
        If a list with two elements [name, value], sets globals()[name] = value.
        Default is None.

    Returns
    -------
    dict or None
        The globals dictionary if x is None, otherwise None.

    Notes
    -----
    This function is designed to be redefined in modules where the
    dtype/device decorators are used as context managers. It provides
    a way to access and modify the global namespace for dynamic
    decorator application.

    Warning
    -------
    This function modifies global state and should be used with caution.
    It's primarily intended for internal use with context manager decorators.
    """
    # define and assign this function to
    # ivy.func_wrapper.globals_getter_func in the module
    # where you want to use the decorators as a context
    # manager
    if not x:
        return globals()
    else:
        globals()[x[0]] = x[1]


class with_unsupported_dtypes(contextlib.ContextDecorator):
    """
    Context manager and decorator for specifying unsupported dtypes.

    This class can be used both as a decorator and as a context manager to
    specify which data types are not supported by a function or code block.
    When used as a context manager, it automatically applies the restriction
    to all functions defined within the context.

    Parameters
    ----------
    *args
        Arguments passed to the dtype restriction system, typically including
        a version dictionary and backend version.
    **kwargs
        Keyword arguments for the dtype restriction system.

    Examples
    --------
    As a decorator:

    >>> @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "numpy")
    ... def my_function(x):
    ...     return x + 1

    As a context manager:

    >>> with with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "numpy"):
    ...     def my_function(x):
    ...         return x + 1

    Notes
    -----
    When used as a context manager, the class maintains a snapshot of the
    global namespace and applies the dtype restrictions to any new functions
    defined within the context.

    The dtype restrictions are applied using the _dtype_device_wrapper_creator
    system, which integrates with Ivy's casting mode operations.
    """
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
    """
    Context manager and decorator for specifying supported dtypes.

    This class can be used both as a decorator and as a context manager to
    specify which data types are supported by a function or code block.
    Functions will only accept the specified dtypes and restrict others.

    Parameters
    ----------
    *args
        Arguments passed to the dtype restriction system, typically including
        a version dictionary and backend version.
    **kwargs
        Keyword arguments for the dtype restriction system.

    Examples
    --------
    As a decorator:

    >>> @with_supported_dtypes({"1.11.0 and above": ("float32", "float64")}, "numpy")
    ... def my_function(x):
    ...     return x + 1

    As a context manager:

    >>> with with_supported_dtypes({"1.11.0 and above": ("float32",)}, "numpy"):
    ...     def my_function(x):
    ...         return x + 1

    Notes
    -----
    This is the complement of with_unsupported_dtypes. When supported dtypes
    are specified, all other dtypes become implicitly unsupported.
    """
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
    """
    Context manager and decorator for specifying unsupported devices.

    This class can be used both as a decorator and as a context manager to
    specify which devices are not supported by a function or code block.
    Arrays on unsupported devices will be moved or cause errors.

    Parameters
    ----------
    *args
        Arguments passed to the device restriction system, typically including
        a version dictionary and backend version.
    **kwargs
        Keyword arguments for the device restriction system.

    Examples
    --------
    As a decorator:

    >>> @with_unsupported_devices({"1.11.0 and below": ("gpu",)}, "numpy")
    ... def my_function(x):
    ...     return x + 1

    As a context manager:

    >>> with with_unsupported_devices({"1.11.0 and below": ("tpu",)}, "jax"):
    ...     def my_function(x):
    ...         return x + 1

    Notes
    -----
    Device restrictions help ensure functions are only called with arrays
    on supported devices, preventing runtime errors and ensuring optimal
    performance.
    """
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
    """
    Context manager/decorator to specify **supported** devices for a function.

    This is the logical inverse of :class:`with_unsupported_devices`.  When used
    the decorated function will *only* accept arrays that reside on the listed
    devices.  Passing arrays from any other device will raise an
    :class:`ivy.utils.exceptions.IvyException` (or enable automatic device
    handling when *soft device* mode is active).
    """
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
    """
    Specify **unsupported (device, dtype)** combinations for a function.

    The accepted argument is a *nested mapping* ``{version: {device: (dtypes,)}}``
    in the same format used throughout the wrapper module.  When an array with
    a disallowed *(device, dtype)* tuple is passed to the decorated function, a
    runtime error will be raised **unless** one of the casting modes is enabled
    that can automatically convert to a supported dtype.
    """
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
                    nested_dic[nested_key] = tuple(dicti[key][nested_key])
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
    """
    Specify **supported (device, dtype)** combinations for a function.

    This class mirrors :class:`with_unsupported_device_and_dtypes` but marks the
    provided combinations as *allowed* - everything else becomes implicitly
    unsupported.
    """
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
                    nested_dic[nested_key] = tuple(dicti[key][nested_key])
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
    """
    Context manager and decorator to override dtype/device restrictions.

    This class marks functions to bypass normal dtype or device restrictions
    that would otherwise be applied by the wrapper system. It's useful for
    special functions that need to handle all dtypes/devices regardless of
    backend limitations.

    Examples
    --------
    As a decorator:

    >>> @override
    ... def my_special_function(x):
    ...     # This function will bypass dtype restrictions
    ...     return x + 1

    As a context manager:

    >>> with override:
    ...     def my_function(x):
    ...         # This function will bypass restrictions
    ...         return x + 1

    Notes
    -----
    Functions marked with override will not have automatic dtype casting
    applied and must handle unsupported types manually. Use with caution
    as it bypasses Ivy's safety mechanisms.
    """
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
