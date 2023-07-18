# global
import ast
import logging
import inspect
import math
from numbers import Number
from typing import Union, Tuple, List, Optional, Callable, Iterable, Any
import numpy as np
import importlib

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_arrays,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_ivy_arrays,
    inputs_to_native_shapes,
    handle_device_shifting,
    inputs_to_native_shapes,
)
from ivy.utils.exceptions import handle_exceptions


# Helpers #
# --------#


def _is_valid_dtypes_attributes(fn: Callable) -> bool:
    if hasattr(fn, "supported_dtypes") and hasattr(fn, "unsupported_dtypes"):
        fn_supported_dtypes = fn.supported_dtypes
        fn_unsupported_dtypes = fn.unsupported_dtypes
        if isinstance(fn_supported_dtypes, dict):
            if isinstance(fn_unsupported_dtypes, dict):
                backend_str = ivy.current_backend_str()
                if (
                    backend_str in fn_supported_dtypes
                    and backend_str in fn_unsupported_dtypes
                ):
                    return False
        else:
            if isinstance(fn_unsupported_dtypes, tuple):
                return False
    return True


def _handle_nestable_dtype_info(fn):
    def _handle_nestable_dtype_info_wrapper(type):
        if isinstance(type, ivy.Container):
            type = type.cont_map(lambda x, kc: fn(x))
            type.__dict__["max"] = type.cont_map(lambda x, kc: x.max)
            type.__dict__["min"] = type.cont_map(lambda x, kc: x.min)
            return type
        return fn(type)

    return _handle_nestable_dtype_info_wrapper


# Unindent every line in the source such that
# class methods can be compiled as normal methods
def _lstrip_lines(source: str) -> str:
    # Separate all lines
    source = source.split("\n")
    # Check amount of indent before first character
    indent = len(source[0]) - len(source[0].lstrip())
    # Remove same spaces from all lines
    for i in range(len(source)):
        source[i] = source[i][indent:]
    source = "\n".join(source)
    return source


# Get the list of function used the function
def _get_function_list(func):
    tree = ast.parse(_lstrip_lines(inspect.getsource(func)))
    names = {}
    # Extract all the call names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nodef = node.func
            if isinstance(nodef, ast.Name):
                names[nodef.id] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )
            elif isinstance(nodef, ast.Attribute):
                if (
                    hasattr(nodef, "value")
                    and hasattr(nodef.value, "id")
                    and nodef.value.id not in ["ivy", "self"]
                ):
                    continue
                names[nodef.attr] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )

    return names


# Get the reference of the functions from string
def _get_functions_from_string(func_names, module):
    ret = set()
    # We only care about the functions in the ivy or the same module
    for func_name in func_names.keys():
        if hasattr(ivy, func_name) and callable(getattr(ivy, func_name, None)):
            ret.add(getattr(ivy, func_name))
        elif hasattr(module, func_name) and callable(getattr(ivy, func_name, None)):
            ret.add(getattr(module, func_name))
        elif callable(getattr(func_names[func_name], func_name, None)):
            ret.add(getattr(func_names[func_name], func_name))
    return ret


# Get dtypes/device of nested functions, used for unsupported and supported dtypes
# IMPORTANT: a few caveats:
# 1. The base functions must be defined in ivy or the same module
# 2. If the dtypes/devices are set not in the base function, it will not be detected
# 3. Nested function cannot be parsed, due to be unable to get function reference
# 4. Functions need to be directly called, not assigned to a variable
def _nested_get(f, base_set, merge_fn, get_fn, wrapper=set):
    visited = set()
    to_visit = [f]
    out = base_set

    while to_visit:
        fn = to_visit.pop()
        if fn in visited:
            continue
        visited.add(fn)

        # if it's in the backend, we can get the dtypes directly
        # if it's in the front end, we need to recurse
        # if it's einops, we need to recurse
        if not getattr(fn, "__module__", None):
            continue
        if "backend" in fn.__module__:
            f_supported = get_fn(fn, False)
            if hasattr(fn, "partial_mixed_handler"):
                f_supported = merge_fn(
                    wrapper(f_supported["compositional"]),
                    wrapper(f_supported["primary"]),
                )
                logging.warning(
                    "This function includes the mixed partial function"
                    f" 'ivy.{fn.__name__}'. Please note that the returned data types"
                    " may not be exhaustive. Please check the dtypes of"
                    f" `ivy.{fn.__name__}` for more details"
                )
            out = merge_fn(wrapper(f_supported), out)
            continue
        elif "frontend" in fn.__module__ or (
            hasattr(fn, "__name__") and "einops" in fn.__name__
        ):
            f_supported = wrapper(get_fn(fn, False))
            out = merge_fn(f_supported, out)

        # skip if it's not a function

        if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            continue

        fl = _get_function_list(fn)
        res = _get_functions_from_string(fl, __import__(fn.__module__))
        to_visit.extend(res)

    return out


# Get the list of dtypes supported by the function
# by default returns the supported dtypes
def _get_dtypes(fn, complement=True):
    supported = set(ivy.valid_dtypes)

    # We only care about getting dtype info from the base function
    # if we do need to at some point use dtype information from the parent function
    # we can comment out the following condition
    is_backend_fn = "backend" in fn.__module__
    is_frontend_fn = "frontend" in fn.__module__
    has_unsupported_dtypes_attr = hasattr(fn, "unsupported_dtypes")
    if not is_backend_fn and not is_frontend_fn and not has_unsupported_dtypes_attr:
        if complement:
            supported = set(ivy.all_dtypes).difference(supported)
        return supported

    # Their values are formatted like either
    # 1. fn.supported_dtypes = ("float16",)
    # Could also have the "all" value for the framework
    basic = [
        ("supported_dtypes", set.intersection, ivy.valid_dtypes),
        ("unsupported_dtypes", set.difference, ivy.invalid_dtypes),
    ]
    for key, merge_fn, base in basic:
        if hasattr(fn, key):
            v = getattr(fn, key)
            # only einops allowed to be a dictionary
            if isinstance(v, dict):
                v = v.get(ivy.current_backend_str(), base)

            ivy.utils.assertions.check_isinstance(v, tuple)
            supported = merge_fn(supported, set(v))

    if complement:
        supported = set(ivy.all_dtypes).difference(supported)

    return tuple(supported)


# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def astype(
    x: Union[ivy.Array, ivy.NativeArray],
    dtype: Union[ivy.Dtype, ivy.NativeDtype],
    /,
    *,
    copy: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Copy an array to a specified data type irrespective of :ref:`type- promotion` rules.

    .. note::
    Casting floating-point ``NaN`` and ``infinity`` values to integral data types
    is not specified and is implementation-dependent.

    .. note::
    When casting a boolean input array to a numeric data type, a value of ``True``
    must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
    to a numeric value equal to ``0``.

    When casting a numeric input array to ``bool``, a value of ``0`` must cast to
    ``False``, and a non-zero value must cast to ``True``.

    Parameters
    ----------
    x
        array to cast.
    dtype
        desired data type.
    copy
        specifies whether to copy an array when the specified ``dtype`` matches
        the data type of the input array ``x``. If ``True``, a newly allocated
        array must always be returned. If ``False`` and the specified ``dtype``
        matches the data type of the input array, the input array must be returned;
        otherwise, a newly allocated must be returned. Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        an array having the specified data type. The returned array must have
        the same shape as ``x``.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2])
    >>> y = ivy.zeros_like(x)
    >>> y = ivy.astype(x, ivy.float64)
    >>> print(y)
    ivy.array([1., 2.])

    >>> x = ivy.array([3.141, 2.718, 1.618])
    >>> ivy.astype(x, ivy.int32, out=y)
    >>> print(y)
    ivy.array([3, 2, 1])

    >>> x = ivy.array([[-1, -2], [0, 2]])
    >>> ivy.astype(x, ivy.float64, out=x)
    >>> print(x)
    ivy.array([[-1., -2.],  [0.,  2.]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([3.141, 2.718, 1.618])
    >>> y = ivy.astype(x, ivy.int32)
    >>> print(y)
    ivy.array([3, 2, 1])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0,2,1]),b=ivy.array([1,0,0]))
    >>> print(ivy.astype(x, ivy.bool))
    {
        a: ivy.array([False, True, True]),
        b: ivy.array([True, False, False])
    }

    Using :class:`ivy.Array` instance method:

    >>> x = ivy.array([[-1, -2], [0, 2]])
    >>> print(x.astype(ivy.float64))
    ivy.array([[-1., -2.],  [0.,  2.]])

    Using :class:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([False,True,True]),
    ...                   b=ivy.array([3.14, 2.718, 1.618]))
    >>> print(x.astype(ivy.int32))
    {
        a: ivy.array([0, 1, 1]),
        b: ivy.array([3, 2, 1])
    }
    """
    return current_backend(x).astype(x, dtype, copy=copy, out=out)


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def broadcast_arrays(*arrays: Union[ivy.Array, ivy.NativeArray]) -> List[ivy.Array]:
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays
        an arbitrary number of arrays to-be broadcasted.

    Returns
    -------
    ret
        A list containing broadcasted arrays of type `ivy.Array`
        Each array must have the same shape, and each array must have the same
        dtype as its corresponding input array.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [ivy.array([1, 2, 3]), ivy.array([4, 5, 6])]

    With :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([0.3, 4.3])
    >>> x2 = ivy.native_array([3.1, 5])
    >>> x3 = ivy.native_array([2, 0])
    >>> y = ivy.broadcast_arrays(x1, x2, x3)
    [ivy.array([0.3, 4.3]), ivy.array([3.1, 5.]), ivy.array([2, 0])]

    With mixed :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([1, 2])
    >>> x2 = ivy.native_array([0.3, 4.3])
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [ivy.array([1, 2]), ivy.array([0.3, 4.3])]

    With :class:`ivy.Container` inputs:

    >>> x1 = ivy.Container(a=ivy.array([3, 1]), b=ivy.zeros(2))
    >>> x2 = ivy.Container(a=ivy.array([4, 5]), b=ivy.array([2, -1]))
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [{
        a: ivy.array([3, 1]),
        b: ivy.array([0., 0.])
    }, {
        a: ivy.array([4, 5]),
        b: ivy.array([2, -1])
    }]

    With mixed :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x1 = ivy.zeros(2)
    >>> x2 = ivy.Container(a=ivy.array([4, 5]), b=ivy.array([2, -1]))
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [{
        a: ivy.array([0., 0.]),
        b: ivy.array([0., 0.])
    }, {
        a: ivy.array([4, 5]),
        b: ivy.array([2, -1])
    }]
    """
    return current_backend(arrays[0]).broadcast_arrays(*arrays)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@inputs_to_native_shapes
@handle_array_function
@handle_device_shifting
def broadcast_to(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    shape: Tuple[int, ...],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Broadcasts an array to a specified shape.

    Parameters
    ----------
    x
        array to broadcast.
    shape
        array shape. Must be compatible with x (see Broadcasting). If
        the array is incompatible with the specified shape, the function
        should raise an exception.
    out
        optional output array, for writing the result to. It must have a
        shape that the inputs broadcast to.

    Returns
    -------
    ret
        an array having a specified shape. Must have the same data type as x.


    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.broadcast_to(x, (3, 3))
    >>> print(y)
    ivy.array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0.1 , 0.3])
    >>> y = ivy.broadcast_to(x, (3, 2))
    >>> print(y)
    ivy.array([[0.1, 0.3],
               [0.1, 0.3],
               [0.1, 0.3]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
    ...                   b=ivy.array([4, 5, 6]))
    >>> y = ivy.broadcast_to(x, (3, 3))
    >>> print(y)
    {
        a: ivy.array([[1, 2, 3],
                      [1, 2, 3],
                      [1, 2, 3]]),
        b: ivy.array([[4, 5, 6],
                      [4, 5, 6],
                      [4, 5, 6]])
    }
    """
    return current_backend(x).broadcast_to(x, shape, out=out)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
def can_cast(
    from_: Union[ivy.Dtype, ivy.Array, ivy.NativeArray],
    to: ivy.Dtype,
    /,
) -> bool:
    """
    Determine if one data type can be cast to another data type according to
    :ref:`type-promotion` rules.

    Parameters
    ----------
    from_
        input data type or array from which to cast.
    to
        desired data type.

    Returns
    -------
    ret
        ``True`` if the cast can occur according to :ref:`type-promotion` rules;
        otherwise, ``False``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.can_cast.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
     With :class:`ivy.Dtype` input:

    >>> print(ivy.can_cast(ivy.uint8, ivy.int32))
    True

    >>> print(ivy.can_cast(ivy.float64, 'int64'))
    False

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> print(ivy.can_cast(x, ivy.float64))
    True

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[-1, -1, -1],
    ...                       [1, 1, 1]],
    ...                       dtype='int16')
    >>> print(ivy.can_cast(x, 'uint8'))
    False

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
    ...                   b=ivy.array([3, 4, 5]))
    >>> print(ivy.can_cast(x, 'int64'))
    {
        a: false,
        b: true
    }
    """
    if isinstance(from_, (ivy.Array, ivy.NativeArray)):
        from_ = from_.dtype
    try:
        ivy.promote_types(from_, to)
        return True
    except KeyError:
        return False


@handle_exceptions
@inputs_to_native_arrays
@handle_device_shifting
def finfo(
    type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray],
    /,
) -> Finfo:
    """
    Machine limits for floating-point data types.

    Parameters
    ----------
    type
        the kind of floating-point data-type about which to get information.

    Returns
    -------
    ret
        an object having the followng attributes:

        - **bits**: *int*

          number of bits occupied by the floating-point data type.

        - **eps**: *float*

          difference between 1.0 and the next smallest representable floating-point
          number larger than 1.0 according to the IEEE-754 standard.

        - **max**: *float*

          largest representable number.

        - **min**: *float*

          smallest representable number.

        - **smallest_normal**: *float*

          smallest positive floating-point number with full precision.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.finfo.html>`_
    in the standard.

    Examples
    --------
    With :class:`ivy.Dtype` input:

    >>> y = ivy.finfo(ivy.float32)
    >>> print(y)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :code:`str` input:

    >>> y = ivy.finfo('float32')
    >>> print(y)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1.3,2.1,3.4], dtype=ivy.float64)
    >>> print(ivy.finfo(x))
    finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
    max=1.7976931348623157e+308, dtype=float64)

    >>> x = ivy.array([0.7,8.4,3.14], dtype=ivy.float16)
    >>> print(ivy.finfo(x))
    finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)

    With :class:`ivy.Container` input:

    >>> c = ivy.Container(x=ivy.array([-9.5,1.8,-8.9], dtype=ivy.float16),
    ...                   y=ivy.array([7.6,8.1,1.6], dtype=ivy.float64))
    >>> print(ivy.finfo(c))
    {
        x: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16),
        y: finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
           max=1.7976931348623157e+308, dtype=float64)
    }
    """
    return current_backend(None).finfo(type)


@handle_exceptions
@inputs_to_native_arrays
@handle_device_shifting
def iinfo(
    type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray],
    /,
) -> Iinfo:
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type
        the kind of integer data-type about which to get information.

    Returns
    -------
    ret
        a class with that encapsules the following attributes:

        - **bits**: *int*

          number of bits occupied by the type.

        - **max**: *int*

          largest representable number.

        - **min**: *int*

          smallest representable number.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.iinfo.html>`_
    in the standard.

    Examples
    --------
    With :class:`ivy.Dtype` input:

    >>> ivy.iinfo(ivy.int32)
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    With :code:`str` input:

    >>> ivy.iinfo('int32')
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    With :class:`ivy.Array` input:

    >>> x = ivy.array([13,21,34], dtype=ivy.int8)
    >>> ivy.iinfo(x)
    iinfo(min=-128, max=127, dtype=int8)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([7,84,314], dtype=ivy.int64)
    >>> ivy.iinfo(x)
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

    With :class:`ivy.Container` input:

    >>> c = ivy.Container(x=ivy.array([0,1800,89], dtype=ivy.uint16),
    ...                   y=ivy.array([76,81,16], dtype=ivy.uint32))
    >>> ivy.iinfo(c)
    {
        x: iinfo(min=0, max=65535, dtype=uint16),
        y: iinfo(min=0, max=4294967295, dtype=uint32)
    }
    """
    return current_backend(None).iinfo(type)


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_device_shifting
def result_type(
    *arrays_and_dtypes: Union[ivy.Array, ivy.NativeArray, ivy.Dtype]
) -> ivy.Dtype:
    """
    Return the dtype that results from applying the type promotion rules (see
    :ref:`type-promotion`) to the arguments.

    .. note::
       If provided mixed dtypes (e.g., integer and floating-point), the returned dtype
       will be implementation-specific.

    Parameters
    ----------
    arrays_and_dtypes
        an arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    ret
        the dtype resulting from an operation involving the input arrays and dtypes.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.result_type.html>`_
    in the standard.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([3, 4, 5])
    >>> y = ivy.array([3., 4., 5.])
    >>> d = ivy.result_type(x, y)
    >>> print(d)
    float32

    With :class:`ivy.Dtype` input:

    >>> d = ivy.result_type(ivy.uint8, ivy.uint64)
    >>> print(d)
    uint64

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array([3, 4, 5]))
    >>> d = x.a.dtype
    >>> print(d)
    int32

    >>> x = ivy.Container(a = ivy.array([3, 4, 5]))
    >>> d = ivy.result_type(x, ivy.float64)
    >>> print(d)
    {
        a: float64
    }
    """
    return current_backend(arrays_and_dtypes[0]).result_type(arrays_and_dtypes)


# Extra #
# ------#

default_dtype_stack = list()
default_float_dtype_stack = list()
default_int_dtype_stack = list()
default_uint_dtype_stack = list()
default_complex_dtype_stack = list()


class DefaultDtype:
    """Ivy's DefaultDtype class."""

    def __init__(self, dtype: ivy.Dtype):
        self._dtype = dtype

    def __enter__(self):
        set_default_dtype(self._dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_dtype()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


class DefaultFloatDtype:
    """Ivy's DefaultFloatDtype class."""

    def __init__(self, float_dtype: ivy.Dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_float_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_float_dtype()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


class DefaultIntDtype:
    """Ivy's DefaultIntDtype class."""

    def __init__(self, int_dtype: ivy.Dtype):
        self._int_dtype = int_dtype

    def __enter__(self):
        set_default_int_dtype(self._int_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


class DefaultUintDtype:
    """Ivy's DefaultUintDtype class."""

    def __init__(self, uint_dtype: ivy.UintDtype):
        self._uint_dtype = uint_dtype

    def __enter__(self):
        set_default_uint_dtype(self._uint_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_uint_dtype()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


class DefaultComplexDtype:
    """Ivy's DefaultComplexDtype class."""

    def __init__(self, complex_dtype: ivy.Dtype):
        self._complex_dtype = complex_dtype

    def __enter__(self):
        set_default_complex_dtype(self._complex_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_complex_dtype()
        if self and (exc_type is not None):
            print(exc_tb)
            raise exc_val
        return self


@handle_exceptions
def dtype_bits(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str], /) -> int:
    """
    Get the number of bits used for representing the input data type.

    Parameters
    ----------
    dtype_in
        The data type to determine the number of bits for.

    Returns
    -------
    ret
        The number of bits used to represent the data type.

    Examples
    --------
    With :class:`ivy.Dtype` inputs:

    >>> x = ivy.dtype_bits(ivy.float32)
    >>> print(x)
    32

    >>> x = ivy.dtype_bits('int64')
    >>> print(x)
    64

    With :class:`ivy.NativeDtype` inputs:

    >>> x = ivy.dtype_bits(ivy.native_bool)
    >>> print(x)
    1
    """
    return current_backend(dtype_in).dtype_bits(dtype_in)


@handle_exceptions
def is_hashable_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype], /) -> bool:
    """
    Check if the given data type is hashable or not.

    Parameters
    ----------
    dtype_in
        The data type to check.

    Returns
    -------
    ret
        True if data type is hashable else False
    """
    try:
        hash(dtype_in)
        return True
    except TypeError:
        return False


@handle_exceptions
def as_ivy_dtype(dtype_in: Union[ivy.Dtype, str], /) -> ivy.Dtype:
    """
    Convert native data type to string representation.

    Parameters
    ----------
    dtype_in
        The data type to convert to string.

    Returns
    -------
    ret
        data type string 'float32'
    """
    return current_backend(None).as_ivy_dtype(dtype_in)


@handle_exceptions
def as_native_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype], /) -> ivy.NativeDtype:
    """
    Convert data type string representation to native data type.

    Parameters
    ----------
    dtype_in
        The data type string to convert to native data type.

    Returns
    -------
    ret
        data type e.g. ivy.float32.
    """
    return current_backend(None).as_native_dtype(dtype_in)


def _check_float64(input) -> bool:
    if ivy.is_array(input):
        return ivy.dtype(input) == "float64"
    if math.isfinite(input):
        m, e = math.frexp(input)
        return (abs(input) > 3.4028235e38) or (e < -126) or (e > 128)
    return False


def _check_complex128(input) -> bool:
    if ivy.is_array(input):
        return ivy.dtype(input) == "complex128"
    elif isinstance(input, np.ndarray):
        return str(input.dtype) == "complex128"
    if hasattr(input, "real") and hasattr(input, "imag"):
        return _check_float64(input.real) and _check_float64(input.imag)
    return False


@handle_exceptions
def closest_valid_dtype(type: Union[ivy.Dtype, str, None], /) -> Union[ivy.Dtype, str]:
    """
    Determine the closest valid datatype to the datatype passed as input.

    Parameters
    ----------
    type
        The data type for which to check the closest valid type for.

    Returns
    -------
    ret
        The closest valid data type as a native ivy.Dtype

    Examples
    --------
    With :class:`ivy.Dtype` input:

    >>> xType = ivy.float16
    >>> yType = ivy.closest_valid_dtype(xType)
    >>> print(yType)
    float16

    With :class:`ivy.NativeDtype` inputs:

    >>> xType = ivy.native_uint16
    >>> yType = ivy.closest_valid_dtype(xType)
    >>> print(yType)
    <dtype:'uint16'>

    With :code:`str` input:

    >>> xType = 'int32'
    >>> yType = ivy.closest_valid_dtype(xType)
    >>> print(yType)
    int32
    """
    return current_backend(type).closest_valid_dtype(type)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
def default_float_dtype(
    *,
    input: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    float_dtype: Optional[Union[ivy.FloatDtype, ivy.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[ivy.Dtype, str, ivy.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the float dtype.
    float_dtype
        The float dtype to be returned.
    as_native
        Whether to return the float dtype as native dtype.

    Returns
    -------
        Return ``float_dtype`` as native or ivy dtype if provided, else
        if ``input`` is given, return its float dtype, otherwise return the
        global default float dtype.

    Examples
    --------
    >>> ivy.default_float_dtype()
    'float32'

    >>> ivy.set_default_float_dtype(ivy.FloatDtype("float64"))
    >>> ivy.default_float_dtype()
    'float64'

    >>> ivy.default_float_dtype(float_dtype=ivy.FloatDtype("float16"))
    'float16'

    >>> ivy.default_float_dtype(input=4294.967346)
    'float32'

    >>> x = ivy.array([9.8,8.9], dtype="float16")
    >>> ivy.default_float_dtype(input=x)
    'float16'
    """
    global default_float_dtype_stack
    if ivy.exists(float_dtype):
        if as_native is True:
            return ivy.as_native_dtype(float_dtype)
        return ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_argwhere(
                input, lambda x: _check_float64(x), stop_after_n_found=1
            ):
                ret = ivy.float64
            else:
                if not default_float_dtype_stack:
                    def_dtype = default_dtype()
                    if ivy.is_float_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "float32"
                else:
                    ret = default_float_dtype_stack[-1]
        elif isinstance(input, Number):
            if _check_float64(input):
                ret = ivy.float64
            else:
                if not default_float_dtype_stack:
                    def_dtype = default_dtype()
                    if ivy.is_float_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "float32"
                else:
                    ret = default_float_dtype_stack[-1]
    else:
        if not default_float_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_float_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "float32"
        else:
            ret = default_float_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.FloatDtype(ivy.as_ivy_dtype(ret))


@handle_exceptions
def infer_default_dtype(
    dtype: Union[ivy.Dtype, ivy.NativeDtype, str], as_native: bool = False
) -> Union[ivy.Dtype, ivy.NativeDtype]:
    """
    Summary.

    Parameters
    ----------
    dtype

    as_native
        (Default value = False)

    Returns
    -------
        Return the default data type for the “kind” (integer or floating-point) of dtype

    Examples
    --------
    >>> ivy.set_default_int_dtype("int32")
    >>> ivy.infer_default_dtype("int8")
    'int8'

    >>> ivy.set_default_float_dtype("float64")
    >>> ivy.infer_default_dtype("float32")
    'float64'

    >>> ivy.set_default_uint_dtype("uint32")
    >>> x = ivy.array([0], dtype="uint64")
    >>> ivy.infer_default_dtype(x.dtype)
    'uint32'
    """
    if ivy.is_complex_dtype(dtype):
        default_dtype = ivy.default_complex_dtype(as_native=as_native)
    elif ivy.is_float_dtype(dtype):
        default_dtype = ivy.default_float_dtype(as_native=as_native)
    elif ivy.is_uint_dtype(dtype):
        default_dtype = ivy.default_uint_dtype(as_native=as_native)
    elif ivy.is_int_dtype(dtype):
        default_dtype = ivy.default_int_dtype(as_native=as_native)
    elif as_native:
        default_dtype = ivy.as_native_dtype("bool")
    else:
        default_dtype = ivy.as_ivy_dtype("bool")
    return default_dtype


@handle_exceptions
@inputs_to_ivy_arrays
def default_dtype(
    *,
    dtype: Optional[Union[ivy.Dtype, str]] = None,
    item: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    as_native: bool = False,
) -> Union[ivy.Dtype, ivy.NativeDtype, str]:
    """
    Parameters
    ----------
    item
        Number or array for inferring the dtype.
    dtype
        The dtype to be returned.
    as_native
        Whether to return the dtype as native dtype.

    Returns
    -------
        Return ``dtype`` as native or ivy dtype if provided, else
        if ``item`` is given, return its dtype, otherwise return the
        global default dtype.

    Examples
    --------
    >>> ivy.default_dtype()
    'float32'

    >>> ivy.set_default_dtype(ivy.bool)
    >>> ivy.default_dtype()
    'bool'

    >>> ivy.set_default_dtype(ivy.int16)
    >>> ivy.default_dtype()
    'int16'

    >>> ivy.set_default_dtype(ivy.float64)
    >>> ivy.default_dtype()
    'float64'

    >>> ivy.default_dtype(dtype="int32")
    'int32'

    >>> ivy.default_dtype(dtype=ivy.float16)
    'float16'

    >>> ivy.default_dtype(item=53.234)
    'float64'

    >>> ivy.default_dtype(item=[1, 2, 3])
    'int32'

    >>> x = ivy.array([5.2, 9.7], dtype="complex128")
    >>> ivy.default_dtype(item=x)
    'complex128'
    """
    if ivy.exists(dtype):
        if as_native is True:
            return ivy.as_native_dtype(dtype)
        return ivy.as_ivy_dtype(dtype)
    as_native = ivy.default(as_native, False)
    if ivy.exists(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_complex_dtype(item):
            return ivy.default_complex_dtype(input=item, as_native=as_native)
        elif ivy.is_float_dtype(item):
            return ivy.default_float_dtype(input=item, as_native=as_native)
        elif ivy.is_uint_dtype(item):
            return ivy.default_int_dtype(input=item, as_native=as_native)
        elif ivy.is_int_dtype(item):
            return ivy.default_int_dtype(input=item, as_native=as_native)
        elif as_native:
            return ivy.as_native_dtype("bool")
        else:
            return "bool"
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.as_ivy_dtype(ret)


@handle_exceptions
@inputs_to_ivy_arrays
def default_int_dtype(
    *,
    input: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    int_dtype: Optional[Union[ivy.IntDtype, ivy.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[ivy.IntDtype, ivy.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the int dtype.
    int_dtype
        The int dtype to be returned.
    as_native
        Whether to return the int dtype as native dtype.

    Returns
    -------
        Return ``int_dtype`` as native or ivy dtype if provided, else
        if ``input`` is given, return its int dtype, otherwise return the
        global default int dtype.

    Examples
    --------
    >>> ivy.set_default_int_dtype(ivy.intDtype("int16"))
    >>> ivy.default_int_dtype()
    'int16'

    >>> ivy.default_int_dtype(input=4294967346)
    'int64'

    >>> ivy.default_int_dtype(int_dtype=ivy.intDtype("int8"))
    'int8'

    >>> x = ivy.array([9,8], dtype="int32")
    >>> ivy.default_int_dtype(input=x)
    'int32'
    """
    global default_int_dtype_stack
    if ivy.exists(int_dtype):
        if as_native is True:
            return ivy.as_native_dtype(int_dtype)
        return ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_argwhere(
                input,
                lambda x: (
                    ivy.dtype(x) == "uint64"
                    if ivy.is_array(x)
                    else x > 9223372036854775807 and x != ivy.inf
                ),
                stop_after_n_found=1,
            ):
                ret = ivy.uint64
            elif ivy.nested_argwhere(
                input,
                lambda x: (
                    ivy.dtype(x) == "int64"
                    if ivy.is_array(x)
                    else x > 2147483647 and x != ivy.inf
                ),
                stop_after_n_found=1,
            ):
                ret = ivy.int64
            else:
                if not default_int_dtype_stack:
                    def_dtype = ivy.default_dtype()
                    if ivy.is_int_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "int32"
                else:
                    ret = default_int_dtype_stack[-1]
        elif isinstance(input, Number):
            if (
                input > 9223372036854775807
                and input != ivy.inf
                and ivy.backend != "torch"
            ):
                ret = ivy.uint64
            elif input > 2147483647 and input != ivy.inf:
                ret = ivy.int64
            else:
                if not default_int_dtype_stack:
                    def_dtype = ivy.default_dtype()
                    if ivy.is_int_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "int32"
                else:
                    ret = default_int_dtype_stack[-1]
    else:
        if not default_int_dtype_stack:
            def_dtype = ivy.default_dtype()
            if ivy.is_int_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "int32"
        else:
            ret = default_int_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.IntDtype(ivy.as_ivy_dtype(ret))


@handle_exceptions
@inputs_to_ivy_arrays
def default_uint_dtype(
    *,
    input: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    uint_dtype: Optional[Union[ivy.UintDtype, ivy.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[ivy.UintDtype, ivy.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the uint dtype.
    uint_dtype
        The uint dtype to be returned.
    as_native
        Whether to return the uint dtype as native dtype.

    Returns
    -------
        Return ``uint_dtype`` as native or ivy dtype if provided, else
        if ``input`` is given, return its uint dtype, otherwise return the
        global default uint dtype.

    Examples
    --------
    >>> ivy.set_default_uint_dtype(ivy.UintDtype("uint16"))
    >>> ivy.default_uint_dtype()
    'uint16'

    >>> ivy.default_uint_dtype(input=4294967346)
    'uint64'

    >>> ivy.default_uint_dtype(uint_dtype=ivy.UintDtype("uint8"))
    'uint8'

    >>> x = ivy.array([9,8], dtype="uint32")
    >>> ivy.default_uint_dtype(input=x)
    'uint32'
    """
    global default_uint_dtype_stack
    if ivy.exists(uint_dtype):
        if as_native is True:
            return ivy.as_native_dtype(uint_dtype)
        return ivy.UintDtype(ivy.as_ivy_dtype(uint_dtype))
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):
            is_native = lambda x: ivy.is_native_array(x)
            if ivy.nested_argwhere(
                input,
                lambda x: (
                    ivy.dtype(x) == "uint64"
                    if is_native(x)
                    else x > 9223372036854775807 and x != ivy.inf
                ),
                stop_after_n_found=1,
            ):
                ret = ivy.uint64
            else:
                if default_uint_dtype_stack:
                    ret = default_uint_dtype_stack[-1]
                else:
                    def_dtype = ivy.default_dtype()
                    if ivy.is_uint_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "uint32"
        elif isinstance(input, Number):
            if input > 4294967295 and input != ivy.inf and ivy.backend != "torch":
                ret = ivy.uint64
            else:
                if default_uint_dtype_stack:
                    ret = default_uint_dtype_stack[-1]
                else:
                    def_dtype = ivy.default_dtype()
                    if ivy.is_uint_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "uint32"
    else:
        if default_uint_dtype_stack:
            ret = default_uint_dtype_stack[-1]
        else:
            def_dtype = ivy.default_dtype()
            if ivy.is_uint_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "uint32"
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.UintDtype(ivy.as_ivy_dtype(ret))


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
@handle_device_shifting
def default_complex_dtype(
    *,
    input: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    complex_dtype: Optional[Union[ivy.ComplexDtype, ivy.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[ivy.Dtype, str, ivy.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the complex dtype.
    complex_dtype
        The complex dtype to be returned.
    as_native
        Whether to return the complex dtype as native dtype.

    Returns
    -------
        Return ``complex_dtype`` as native or ivy dtype if provided, else
        if ``input`` is given, return its complex dtype, otherwise return the
        global default complex dtype.

    Examples
    --------
    >>> ivy.default_complex_dtype()
    'complex64'

    >>> ivy.set_default_complex_dtype(ivy.ComplexDtype("complex64"))
    >>> ivy.default_complex_dtype()
    'complex64'

    >>> ivy.default_complex_dtype(complex_dtype=ivy.ComplexDtype("complex128"))
    'complex128'

    >>> ivy.default_complex_dtype(input=4294.967346)
    'complex64'

    >>> x = ivy.array([9.8,8.9], dtype="complex128")
    >>> ivy.default_complex_dtype(input=x)
    'complex128'
    """
    global default_complex_dtype_stack
    if ivy.exists(complex_dtype):
        if as_native is True:
            return ivy.as_native_dtype(complex_dtype)
        return ivy.ComplexDtype(ivy.as_ivy_dtype(complex_dtype))
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_argwhere(
                input, lambda x: _check_complex128(x), stop_after_n_found=1
            ):
                ret = ivy.complex128
            else:
                if not default_complex_dtype_stack:
                    def_dtype = default_dtype()
                    if ivy.is_complex_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "complex64"
                else:
                    ret = default_complex_dtype_stack[-1]
        elif isinstance(input, Number):
            if _check_complex128(input):
                ret = ivy.complex128
            else:
                if not default_complex_dtype_stack:
                    def_dtype = default_dtype()
                    if ivy.is_complex_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "complex64"
                else:
                    ret = default_complex_dtype_stack[-1]
    else:
        if not default_complex_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_complex_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "complex64"
        else:
            ret = default_complex_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.ComplexDtype(ivy.as_ivy_dtype(ret))


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_device_shifting
def dtype(
    x: Union[ivy.Array, ivy.NativeArray], *, as_native: bool = False
) -> Union[ivy.Dtype, ivy.NativeDtype]:
    """
    Get the data type for input array x.

    Parameters
    ----------
    x
        Tensor for which to get the data type.
    as_native
        Whether or not to return the dtype in string format. Default is ``False``.

    Returns
    -------
    ret
        Data type of the array.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x1 = ivy.array([1.0, 2.0, 3.5, 4.5, 5, 6])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    float32

    With :class:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([1, 0, 1, -1, 0])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    int32

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.native_array([1.0, 2.0, -1.0, 4.0, 1.0]),
    ...                   b=ivy.native_array([1, 0, 0, 0, 1]))
    >>> y = ivy.dtype(x.a)
    >>> print(y)
    float32
    """
    return current_backend(x).dtype(x, as_native=as_native)


@handle_exceptions
@handle_nestable
def function_supported_dtypes(fn: Callable, recurse: bool = True) -> Union[Tuple, dict]:
    """
    Return the supported data types of the current backend's function. The function
    returns a dict containing the supported dtypes for the compositional and primary
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the supported dtype attribute
    recurse
        Whether to recurse into used ivy functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the supported dtypes of the function

    Examples
    --------
    >>> print(ivy.function_supported_dtypes(ivy.acosh))
    ('bool', 'float64', 'int64', 'uint8', 'int8', 'float32', 'int32', 'int16', \
    'bfloat16')
    """
    ivy.utils.assertions.check_true(
        _is_valid_dtypes_attributes(fn),
        (
            "supported_dtypes and unsupported_dtypes attributes cannot both exist "
            "in a particular backend"
        ),
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_supported_dtypes(fn.compos, recurse=recurse),
            "primary": _get_dtypes(fn, complement=False),
        }
    else:
        supported_dtypes = set(_get_dtypes(fn, complement=False))
        if recurse:
            supported_dtypes = _nested_get(
                fn, supported_dtypes, set.intersection, function_supported_dtypes
            )
    return (
        supported_dtypes
        if isinstance(supported_dtypes, dict)
        else tuple(supported_dtypes)
    )


@handle_exceptions
@handle_nestable
def function_unsupported_dtypes(
    fn: Callable, recurse: bool = True
) -> Union[Tuple, dict]:
    """
    Return the unsupported data types of the current backend's function. The function
    returns a dict containing the unsupported dtypes for the compositional and primary
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the unsupported dtype attribute
    recurse
        Whether to recurse into used ivy functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the unsupported dtypes of the function

    Examples
    --------
    >>> ivy.set_backend('torch')
    >>> print(ivy.function_unsupported_dtypes(ivy.acosh))
    ('float16','uint16','uint32','uint64')
    """
    ivy.utils.assertions.check_true(
        _is_valid_dtypes_attributes(fn),
        (
            "supported_dtypes and unsupported_dtypes attributes cannot both exist "
            "in a particular backend"
        ),
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_unsupported_dtypes(fn.compos, recurse=recurse),
            "primary": _get_dtypes(fn, complement=True),
        }
    else:
        unsupported_dtypes = set(_get_dtypes(fn, complement=True))
        if recurse:
            unsupported_dtypes = _nested_get(
                fn, unsupported_dtypes, set.union, function_unsupported_dtypes
            )

    return (
        unsupported_dtypes
        if isinstance(unsupported_dtypes, dict)
        else tuple(unsupported_dtypes)
    )


@handle_exceptions
def invalid_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str, None], /) -> bool:
    """
    Determine whether the provided data type is not support by the current framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend non-support

    Returns
    -------
    ret
        Boolean, whether the data-type string is un-supported.

    Examples
    --------
    >>> print(ivy.invalid_dtype(None))
    False

    >>> print(ivy.invalid_dtype("uint64"))
    False

    >>> print(ivy.invalid_dtype(ivy.float64))
    False

    >>> print(ivy.invalid_dtype(ivy.native_uint8))
    False
    """
    if dtype_in is None:
        return False
    return ivy.as_ivy_dtype(dtype_in) in ivy.invalid_dtypes


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_bool_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """
    Determine whether the input data type is a bool data type.

    Parameters
    ----------
    dtype_in
        input data type to test.

    Returns
    -------
    ret
        "True" if the input data type is a bool, otherwise "False".

    Both the description and the type hints above assumes an array input for
    simplicity but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "bool" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return (
            True
            if isinstance(dtype_in, (bool, np.bool)) and not isinstance(dtype_in, bool)
            else False
        )
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_argwhere(
                dtype_in,
                lambda x: isinstance(x, (bool, np.bool)) and not type(x) == int,
            )
            else False
        )
    return "bool" in ivy.as_ivy_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_int_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """
    Determine whether the input data type is an int data type.

    Parameters
    ----------
    dtype_in
        input data type to test.

    Returns
    -------
    ret
        "True" if the input data type is an integer, otherwise "False".

    Both the description and the type hints above assumes an array input for
    simplicity but this function is *nestable*, and therefore also accepts
    :class:`ivy.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Dtype` input:

    >>> x = ivy.is_int_dtype(ivy.float64)
    >>> print(x)
    False

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> x.dtype
    float32
    >>> print(ivy.is_int_dtype(x))
    False

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[-1, -1, -1], [1, 1, 1]], dtype=ivy.int16)
    >>> print(ivy.is_int_dtype(x))
    True

    With :code:`Number` input:

    >>> x = 1
    >>> print(ivy.is_int_dtype(x))
    True

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),b=ivy.array([3, 4, 5]))
    >>> x.a.dtype
    float32
    >>> x.b.dtype
    int32
    >>> print(ivy.is_int_dtype(x))
    {
        a: false,
        b: true
    }
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return (
            True
            if isinstance(dtype_in, (int, np.integer))
            and not isinstance(dtype_in, bool)
            else False
        )
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_argwhere(
                dtype_in,
                lambda x: (
                    isinstance(x, (int, np.integer))
                    or (ivy.is_array(x) and "int" in ivy.dtype(x))
                )
                and not type(x) == bool,
            )
            else False
        )
    return "int" in ivy.as_ivy_dtype(dtype_in)


@handle_exceptions
def check_float(x: Any) -> bool:
    """
    Check if the input is a float or a float-like object.

    Parameters
    ----------
    x
        Input to check.

    Returns
    -------
    ret
        "True" if the input is a float or a float-like object, otherwise "False".
    """
    return isinstance(x, (int, np.float)) and not type(x) == bool


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_float_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """
    Determine whether the input data type is a float dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a floating point dtype

    Examples
    --------
    >>> x = ivy.is_float_dtype(ivy.float32)
    >>> print(x)
    True

    >>> arr = ivy.array([1.2, 3.2, 4.3], dtype=ivy.float32)
    >>> print(ivy.is_float_dtype(arr))
    True
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return True if isinstance(dtype_in, (float, np.floating)) else False
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_argwhere(
                dtype_in,
                lambda x: isinstance(x, (float, np.floating))
                or (ivy.is_array(x) and "float" in ivy.dtype(x)),
            )
            else False
        )
    return "float" in as_ivy_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_uint_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """
    Determine whether the input data type is a uint dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a uint dtype

    Examples
    --------
    >>> ivy.is_uint_dtype(ivy.UintDtype("uint16"))
    True

    >>> ivy.is_uint_dtype(ivy.Dtype("uint8"))
    True

    >>> ivy.is_uint_dtype(ivy.IntDtype("int64"))
    False
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "uint" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, np.unsignedinteger)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return ivy.nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, np.unsignedinteger)
            or (ivy.is_array(x) and "uint" in ivy.dtype(x)),
        )
    return "uint" in as_ivy_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_ivy_arrays
def is_complex_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """
    Determine whether the input data type is a complex dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a complex dtype

    Examples
    --------
    >>> ivy.is_complex_dtype(ivy.ComplexDtype("complex64"))
    True

    >>> ivy.is_complex_dtype(ivy.Dtype("complex128"))
    True

    >>> ivy.is_complex_dtype(ivy.IntDtype("int64"))
    False
    """
    if ivy.is_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "complex" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (complex, np.complexfloating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return ivy.nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, (complex, np.complexfloating))
            or (ivy.is_array(x) and "complex" in ivy.dtype(x)),
        )
    return "complex" in as_ivy_dtype(dtype_in)


@handle_exceptions
def promote_types(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
    *,
    array_api_promotion: bool = False,
) -> ivy.Dtype:
    """
    Promote the datatypes type1 and type2, returning the data type they promote to.

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote
    array_api_promotion
        whether to only use the array api promotion rules

    Returns
    -------
    ret
        The type that both input types promote to
    """
    query = [ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2)]
    query.sort(key=lambda x: str(x))
    query = tuple(query)

    def _promote(query):
        if array_api_promotion:
            return ivy.array_api_promotion_table[query]
        return ivy.promotion_table[query]

    try:
        ret = _promote(query)
    except KeyError:
        # try again with the dtypes swapped
        query = (query[1], query[0])
        try:
            ret = _promote(query)
        except KeyError:
            raise ivy.utils.exceptions.IvyDtypePromotionError(
                "these dtypes ({} and {}) are not type promotable, ".format(
                    type1, type2
                )
            )
    return ret


@handle_exceptions
def set_default_dtype(dtype: Union[ivy.Dtype, ivy.NativeDtype, str], /):
    """
    Set the datatype `dtype` as default data type.

    Parameters
    ----------
    dtype
        the data_type to set as default data type

    Examples
    --------
    With :class:`ivy.Dtype` input:

    >>> ivy.set_default_dtype(ivy.bool)
    >>> ivy.default_dtype_stack
    ['bool']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype("float64")
    >>> ivy.default_dtype_stack
    ['float64']
    >>> ivy.unset_default_dtype()

    With :class:`ivy.NativeDtype` input:

    >>> ivy.set_default_dtype(ivy.native_uint64)
    >>> ivy.default_dtype_stack
    ['uint64']
    """
    dtype = ivy.as_ivy_dtype(dtype)
    ivy.utils.assertions._check_jax_x64_flag(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


@handle_exceptions
def set_default_float_dtype(float_dtype: Union[ivy.Dtype, str], /):
    """
    Set the 'float_dtype' as the default data type.

    Parameters
    ----------
    float_dtype
        The float data type to be set as the default.

    Examples
    --------
    With :class: `ivy.Dtype` input:

    >>> ivy.set_default_float_dtype(ivy.floatDtype("float64"))
    >>> ivy.default_float_dtype()
    'float64'

    >>> ivy.set_default_float_dtype(ivy.floatDtype("float32"))
    >>> ivy.default_float_dtype()
    'float32'
    """
    float_dtype = ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
    ivy.utils.assertions._check_jax_x64_flag(float_dtype)
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


@handle_exceptions
def set_default_int_dtype(int_dtype: Union[ivy.Dtype, str], /):
    """
    Set the 'int_dtype' as the default data type.

    Parameters
    ----------
    int_dtype
        The integer data type to be set as the default.

    Examples
    --------
    With :class: `ivy.Dtype` input:

    >>> ivy.set_default_int_dtype(ivy.intDtype("int64"))
    >>> ivy.default_int_dtype()
    'int64'

    >>> ivy.set_default_int_dtype(ivy.intDtype("int32"))
    >>> ivy.default_int_dtype()
    'int32'
    """
    int_dtype = ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
    ivy.utils.assertions._check_jax_x64_flag(int_dtype)
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


@handle_exceptions
def set_default_uint_dtype(uint_dtype: Union[ivy.Dtype, str], /):
    """
    Set the uint dtype to be default.

    Parameters
    ----------
    uint_dtype
        The uint dtype to be set as default.

    Examples
    --------
    >>> ivy.set_default_uint_dtype(ivy.UintDtype("uint8"))
    >>> ivy.default_uint_dtype()
    'uint8'

    >>> ivy.set_default_uint_dtype(ivy.UintDtype("uint64"))
    >>> ivy.default_uint_dtype()
    'uint64'
    """
    uint_dtype = ivy.UintDtype(ivy.as_ivy_dtype(uint_dtype))
    ivy.utils.assertions._check_jax_x64_flag(uint_dtype)
    global default_uint_dtype_stack
    default_uint_dtype_stack.append(uint_dtype)


@handle_exceptions
def set_default_complex_dtype(complex_dtype: Union[ivy.Dtype, str], /):
    """
    Set the 'complex_dtype' as the default data type.

    Parameters
    ----------
    complex_dtype
        The complex data type to be set as the default.

    Examples
    --------
    With :class: `ivy.Dtype` input:

    >>> ivy.set_default_complex_dtype(ivy.ComplexDtype("complex64"))
    >>> ivy.default_complex_dtype()
    'complex64'

    >>> ivy.set_default_float_dtype(ivy.ComplexDtype("complex128"))
    >>> ivy.default_complex_dtype()
    'complex128'
    """
    complex_dtype = ivy.ComplexDtype(ivy.as_ivy_dtype(complex_dtype))
    ivy.utils.assertions._check_jax_x64_flag(complex_dtype)
    global default_complex_dtype_stack
    default_complex_dtype_stack.append(complex_dtype)


@handle_exceptions
def type_promote_arrays(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple:
    """
    Type promote the input arrays, returning new arrays with the shared correct data
    type.

    Parameters
    ----------
    x1
        the first of the two arrays to type promote
    x2
        the second of the two arrays to type promote

    Returns
    -------
    ret1, ret2
        The input arrays after type promotion
    """
    new_type = ivy.promote_types(ivy.dtype(x1), ivy.dtype(x2))
    return ivy.astype(x1, new_type), ivy.astype(x2, new_type)


@handle_exceptions
def unset_default_dtype():
    """
    Reset the current default dtype to the previous state.

    Examples
    --------
    >>> ivy.set_default_dtype(ivy.int32)
    >>> ivy.set_default_dtype(ivy.bool)
    >>> ivy.default_dtype_stack
    ['int32', 'bool']

    >>> ivy.unset_default_dtype()
    >>> ivy.default_dtype_stack
    ['int32']

    >>> ivy.unset_default_dtype()
    >>> ivy.default_dtype_stack
    []
    """
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_float_dtype():
    """
    Reset the current default float dtype to the previous state.

    Examples
    --------
    >>> ivy.set_default_float_dtype(ivy.float32)
    >>> ivy.set_default_float_dtype(ivy.float64)
    >>> ivy.default_float_dtype_stack
    ['float32','float64']

    >>> ivy.unset_default_float_dtype()
    >>> ivy.default_float_dtype_stack
    ['float32']
    """
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_int_dtype():
    """
    Reset the current default int dtype to the previous state.

    Parameters
    ----------
    None-

    Examples
    --------
    >>> ivy.set_default_int_dtype(ivy.intDtype("int16"))
    >>> ivy.default_int_dtype()
    'int16'

    >>> ivy.unset_default_int_dtype()
    >>> ivy.default_int_dtype()
    'int32'
    """
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_uint_dtype():
    """
    Reset the current default uint dtype to the previous state.

    Examples
    --------
    >>> ivy.set_default_uint_dtype(ivy.UintDtype("uint8"))
    >>> ivy.default_uint_dtype()
    'uint8'

    >>> ivy.unset_default_uint_dtype()
    >>> ivy.default_uint_dtype()
    'uint32'
    """
    global default_uint_dtype_stack
    if default_uint_dtype_stack:
        default_uint_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_complex_dtype():
    """
    Reset the current default complex dtype to the previous state.

    Examples
    --------
    >>> ivy.set_default_complex_dtype(ivy.complex64)
    >>> ivy.set_default_complex_dtype(ivy.complex128)
    >>> ivy.default_complex_dtype_stack
    ['complex64','complex128']

    >>> ivy.unset_default_complex_dtype()
    >>> ivy.default_complex_dtype_stack
    ['complex64']
    """
    global default_complex_dtype_stack
    if default_complex_dtype_stack:
        default_complex_dtype_stack.pop(-1)


@handle_exceptions
def valid_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str, None], /) -> bool:
    """
    Determine whether the provided data type is supported by the current framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend support

    Returns
    -------
    ret
        Boolean, whether or not the data-type string is supported.

    Examples
    --------
    >>> print(ivy.valid_dtype(None))
    True

    >>> print(ivy.valid_dtype(ivy.float64))
    True

    >>> print(ivy.valid_dtype('bool'))
    True

    >>> print(ivy.valid_dtype(ivy.native_float16))
    True
    """
    if dtype_in is None:
        return True
    return ivy.as_ivy_dtype(dtype_in) in ivy.valid_dtypes


@handle_exceptions
def promote_types_of_inputs(
    x1: Union[ivy.NativeArray, Number, Iterable[Number]],
    x2: Union[ivy.NativeArray, Number, Iterable[Number]],
    /,
    *,
    array_api_promotion: bool = False,
) -> Tuple[ivy.NativeArray, ivy.NativeArray]:
    """
    Promote the dtype of the given native array inputs to a common dtype based on type
    promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
    """

    def _special_case(a1, a2):
        # check for float number and integer array case
        return isinstance(a1, float) and "int" in str(a2.dtype)

    if hasattr(x1, "dtype") and not hasattr(x2, "dtype"):
        device = ivy.default_device(item=x1, as_native=True)
        if x1.dtype == bool and not isinstance(x2, bool):
            x2 = (
                ivy.asarray(x2, device=device)
                if not _special_case(x2, x1)
                else ivy.asarray(x2, dtype="float64", device=device)
            )
        else:
            x2 = (
                ivy.asarray(x2, dtype=x1.dtype, device=device)
                if not _special_case(x2, x1)
                else ivy.asarray(x2, dtype="float64", device=device)
            )
    elif hasattr(x2, "dtype") and not hasattr(x1, "dtype"):
        device = ivy.default_device(item=x2, as_native=True)
        if x2.dtype == bool and not isinstance(x1, bool):
            x1 = (
                ivy.asarray(x1, device=device)
                if not _special_case(x1, x2)
                else ivy.asarray(x1, dtype="float64", device=device)
            )
        else:
            x1 = (
                ivy.asarray(x1, dtype=x2.dtype, device=device)
                if not _special_case(x1, x2)
                else ivy.asarray(x1, dtype="float64", device=device)
            )
    elif not (hasattr(x1, "dtype") or hasattr(x2, "dtype")):
        x1 = ivy.asarray(x1)
        x2 = ivy.asarray(x2)

    if x1.dtype != x2.dtype:
        promoted = promote_types(
            x1.dtype, x2.dtype, array_api_promotion=array_api_promotion
        )
        x1 = ivy.astype(x1, promoted, copy=False)
        x2 = ivy.astype(x2, promoted, copy=False)

    ivy.utils.assertions._check_jax_x64_flag(x1.dtype)
    return ivy.to_native(x1), ivy.to_native(x2)


@handle_exceptions
def is_native_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype], /) -> bool:
    """
    Determine whether the input dtype is a Native dtype.

    Parameters
    ----------
    dtype_in
        Determine whether the input data type is a native data type object.

    Returns
    -------
    ret
        Boolean, whether or not dtype_in is a native data type.

    Examples
    --------
    >>> ivy.set_backend('numpy')
    >>> ivy.is_native_dtype(np.int32)
    True

    >>> ivy.set_backend('numpy')
    >>> ivy.is_native_array(ivy.float64)
    False
    """
    try:
        return current_backend(None).is_native_dtype(dtype_in)
    except ValueError:
        return False
