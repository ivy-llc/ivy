import contextlib
import ivy
import functools
import logging
import weakref
import warnings
import copy as python_copy
from types import FunctionType
from typing import Callable
import inspect


# for wrapping (sequence matters)
FN_DECORATORS = [
    "infer_device",
    "infer_dtype",
    "handle_array_function",
    "integer_arrays_to_float",
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
    "handle_nestable",
    "handle_exceptions",
    "with_unsupported_dtypes",
    "handle_nans",
    "handle_mixed_function",
]


# Helpers #
# --------#


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
    arr = None
    if args:
        arr_idxs = ivy.nested_argwhere(args, ivy.is_array, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(args, arr_idxs[0])
        else:
            arr_idxs = ivy.nested_argwhere(kwargs, ivy.is_array, stop_after_n_found=1)
            if arr_idxs:
                arr = ivy.index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = ivy.nested_argwhere(kwargs, ivy.is_array, stop_after_n_found=1)
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
    Helper to recursively check if a N-level nested `sequence` contains either a
    `value` or contains a value of type `_type` and return a boolean flag
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


def handle_array_function(func):
    """
    Wrap a function to extract the relevant argument types to be passed to
    array_function method.
    """

    @functools.wraps(func)
    def _handle_array_function(*args, **kwargs):
        overloaded_types = []
        overloaded_args = []

        for arg in args + tuple(kwargs.values()):
            if ivy.exists(arg) and (
                not isinstance(arg, ivy.Container)
                and hasattr(arg, "__ivy_array_function__")
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
            if ivy.exists(arg) and isinstance(arg, ivy.Container):
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
                                if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                    index = i
                                    break
                            overloaded_args.insert(index, arg)

        success, value = try_array_function_override(
            ivy.__dict__[func.__name__], overloaded_args, overloaded_types, args, kwargs
        )
        if success:
            return value
        return func(*args, **kwargs)

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
    def new_fn(*args, **kwargs):
        ivy_shape_idxs = ivy.nested_argwhere(
            [args, kwargs], lambda x: isinstance(x, ivy.Shape)
        )
        ivy_shapes = ivy.multi_index_nest([args, kwargs], ivy_shape_idxs)
        native_shapes = [ivy.to_native_shape(shape) for shape in ivy_shapes]
        args, kwargs = ivy.set_nest_at_indices(
            [args, kwargs], ivy_shape_idxs, native_shapes, shallow=False
        )
        return fn(*args, **kwargs)

    new_fn.inputs_to_native_shapes = True
    return new_fn


def outputs_to_ivy_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        native_shape_idxs = ivy.nested_argwhere(
            [args, kwargs], lambda x: isinstance(x, ivy.NativeShape)
        )
        native_shapes = ivy.multi_index_nest([args, kwargs], native_shape_idxs)
        ivy_shapes = ivy.to_ivy(native_shapes)
        ivy.set_nest_at_indices([args, kwargs], native_shape_idxs, ivy_shapes)
        return fn(*args, **kwargs)

    new_fn.outputs_to_ivy_shapes = True
    return new_fn


def outputs_to_ivy_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
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
        return (
            ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
            if ivy.get_array_mode()
            else ret
        )

    _outputs_to_ivy_arrays.outputs_to_ivy_arrays = True
    return _outputs_to_ivy_arrays


def output_to_native_arrays(fn: Callable) -> Callable:
    """
    Calls the function, and then converts all `ivy.Array` instances in
    the function return into `ivy.NativeArray` instances.

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
    Wraps `fn` so that input arrays are all converted to `ivy.Array` instances
    and return arrays are all converted to `ivy.NativeArray` instances.
    """
    return output_to_native_arrays(inputs_to_ivy_arrays(fn))


def to_native_arrays_and_back(fn: Callable) -> Callable:
    """
    Wraps `fn` so that input arrays are all converted to `ivy.NativeArray` instances
    and return arrays are all converted to `ivy.Array` instances.
    """
    return outputs_to_ivy_arrays(inputs_to_native_arrays(fn))


def handle_view(fn: Callable) -> Callable:
    """
    Wraps `fn` and performs view handling if copy is False. Used for functional
    backends (Jax and TensorFlow). Checks if the first arg is a view or original
    array by checking if the ._base attribute is populated. If it's original
    it adds the returned array to its view references, then the returned array
    adds the operation to its manipulation stack and stores the original as its
    base. If the first arg is a view, then the returned array copies its base and
    manipulation stack, appends the new operation to the manipulation stack and
    appends its reference to the base array's view_refs attribute.
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
    Wraps `fn` and performs view handling specifically for indexing. As with NumPy
    it returns a copy if advanced indexing is performed. Used for functional
    backends (Jax and TensorFlow). Checks if the first arg is a view or
    original array by checking if the ._base attribute is populated. If it's original
    it adds the returned array to its view references, then the returned array
    adds the operation to its manipulation stack and stores the original as its
    base. If the first arg is a view, then the returned array copies its base and
    manipulation stack, appends the new operation to the manipulation stack and
    appends its reference to the base array's view_refs attribute.
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


# Data Type Handling #
# -------------------#


def infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
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
        ivy.utils.assertions._check_jax_x64_flag(dtype)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    _infer_dtype.infer_dtype = True
    return _infer_dtype


def integer_arrays_to_float(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _integer_arrays_to_float(*args, **kwargs):
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

        def _to_float_array(x):
            if not ivy.is_array(x) or not ivy.is_int_dtype(x.dtype):
                return x
            if ivy.is_ivy_array(x):
                return ivy.asarray(x, dtype=ivy.default_float_dtype())
            return ivy.native_array(x, dtype=ivy.default_float_dtype(as_native=True))

        args = ivy.nested_map(args, _to_float_array, to_mutable=True)
        kwargs = ivy.nested_map(kwargs, _to_float_array, to_mutable=True)
        return fn(*args, **kwargs)

    _integer_arrays_to_float.integer_arrays_to_float = True
    return _integer_arrays_to_float


# Device Handling #
# ----------------#


def infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _infer_device(*args, device=None, **kwargs):
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

    _infer_device.infer_device = True
    return _infer_device


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn: Callable) -> Callable:
    handle_out_in_backend = hasattr(fn, "support_native_out")
    handle_out_in_ivy = hasattr(fn, "mixed_function")

    @functools.wraps(fn)
    def _handle_out_argument(*args, out=None, **kwargs):
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
        if out is None or handle_out_in_ivy:
            return fn(*args, out=out, **kwargs)
        if handle_out_in_backend:
            # extract underlying native array for out
            native_out = ivy.to_native(out)
            # compute return, with backend inplace update handled by
            # the backend function
            ret = fn(*args, out=native_out, **kwargs)
            if isinstance(ret, (tuple, list)):
                for i in range(len(ret)):
                    out[i].data = ivy.to_native(ret[i])
                    if ivy.backend == "torch":
                        _update_torch_views(out[i])
            else:
                out.data = ivy.to_native(ret)
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
        if hasattr(ivy.Container, "_static_" + fn_name):
            cont_fn = getattr(ivy.Container, "_static_" + fn_name)
        else:
            cont_fn = lambda *args, **kwargs: ivy.Container.cont_multi_map_in_function(
                fn, *args, **kwargs
            )
        if ivy.get_nestable_mode() and (
            ivy.nested_any(args, ivy.is_ivy_container, check_nests=True)
            or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True)
        ):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    _handle_nestable.handle_nestable = True
    return _handle_nestable


# Functions #


def _wrap_function(
    key: str, to_wrap: Callable, original: Callable, compositional: bool = False
) -> Callable:
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
            if attr.startswith("_") or hasattr(ivy, attr) or attr == "handles_out_arg":
                continue
            setattr(to_wrap, attr, getattr(original, attr))
        # Copy docstring
        docstring_attr = ["__annotations__", "__doc__"]
        for attr in docstring_attr:
            setattr(to_wrap, attr, getattr(original, attr))
        # wrap decorators
        mixed = hasattr(original, "mixed_function")
        if mixed:
            to_replace = {
                True: ["inputs_to_ivy_arrays"],
                False: [
                    "outputs_to_ivy_arrays",
                    "inputs_to_native_arrays",
                ],
            }
            # if the backend has a primary implementation
            # we'll store the compositional fn's reference
            # for the handle_mixed_function decorator
            if to_wrap != original:
                to_wrap.compos = original
            for attr in to_replace[compositional]:
                setattr(original, attr, True)

        for attr in FN_DECORATORS:
            if hasattr(original, attr) and not hasattr(to_wrap, attr):
                to_wrap = getattr(ivy, attr)(to_wrap)
    return to_wrap


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

    # if no version is found, return the last version
    return dic[list(dic.keys())[-1]]


def _versioned_attribute_factory(attribute_function, base):
    class VersionedAttributes(base):
        """
        Creates a class which inherits `base` this way if isinstance is called on an
        instance of the class, it will return True if testing for the baseclass, such as
        isinstance(instance, tuple) if `base` is tuple.
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
    Creates a wrapper for a dtype or device attribute, which returns the correct
    dtype or device for the current version of the backend.

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
        typesets = {
            "valid": ivy.valid_dtypes,
            "numeric": ivy.valid_numeric_dtypes,
            "float": ivy.valid_float_dtypes,
            "integer": ivy.valid_int_dtypes,
            "unsigned": ivy.valid_uint_dtypes,
            "complex": ivy.valid_complex_dtypes,
        }
        for key, value in version_dict.items():
            for i, v in enumerate(value):
                if v in typesets:
                    version_dict[key] = (
                        version_dict[key][:i] + typesets[v] + version_dict[key][i + 1 :]
                    )

        def _wrapped(func):
            val = _versioned_attribute_factory(
                lambda: _dtype_from_version(version_dict, version), t
            )
            if hasattr(func, "override"):
                # we do nothing
                return func
            if not exclusive:
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
                        setattr(func, attrib, val)
                        setattr(func, "dictionary_info", version_dict)
                    elif hasattr(func, "exclusive"):
                        if attrib == attribs:
                            old_version_dict = getattr(func, "dictionary_info")
                            old_version_dict.update(version_dict)
                            val = _versioned_attribute_factory(
                                lambda: _dtype_from_version(version_dict, version), t
                            )
                            setattr(func, attrib, val)
                        else:
                            # for conflicting ones we do nothing
                            pass
            else:
                setattr(func, attrib, val)
                setattr(func, "dictionary_info", version_dict)

            return func

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
        Checks for the existence of nans in all arrays in the `args`
        and `kwargs`. The presence of nans is then handled depending
        on the enabled `nan_policy`.

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
        nan_policy = ivy.get_nan_policy()
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


def handle_mixed_function(condition) -> Callable:
    def inner_function(fn):
        @functools.wraps(fn)
        def _handle_mixed_function(*args, **kwargs):
            compos = getattr(_handle_mixed_function, "compos")
            if condition(*args, **kwargs):
                return fn(*args, **kwargs)

            return compos(*args, **kwargs)

        _handle_mixed_function.handle_mixed_functions = True
        return _handle_mixed_function

    return inner_function


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
            setattr(func, "override")
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
