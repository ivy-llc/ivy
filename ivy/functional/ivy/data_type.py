# global
import math
import importlib
import numpy as np
from numbers import Number
from typing import Union, Tuple, List, Optional

# local
import ivy
from ivy.framework_handler import current_framework as _cur_framework

# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


# Dtype Info #


def can_cast(
    from_: Union[ivy.Dtype, ivy.Array, ivy.NativeArray], to: ivy.Dtype
) -> bool:
    """
    Determines if one data type can be cast to another data type according to
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

    """
    return _cur_framework(from_).can_cast(from_, to)


# noinspection PyShadowingBuiltins
def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Iinfo:
    """Machine limits for integer data types.

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

    """
    return _cur_framework(None).iinfo(type)


# noinspection PyShadowingBuiltins
def finfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Finfo:
    """Machine limits for floating-point data types.

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

    """
    return _cur_framework(None).finfo(type)


def broadcast_to(
    x: Union[ivy.Array, ivy.NativeArray], shape: Tuple[int, ...]
) -> ivy.Array:
    """Broadcasts an array to a specified shape.

    Parameters
    ----------
    x
        array to broadcast.
    shape
        array shape. Must be compatible with x (see Broadcasting). If
        the array is incompatible with the specified shape, the function should raise an
        exception.

    Returns
    -------
    ret
        an array having a specified shape. Must have the same data type as x.

    """
    return _cur_framework(x).broadcast_to(x, shape)


def broadcast_arrays(*arrays: Union[ivy.Array, ivy.NativeArray]) -> List[ivy.Array]:
    """Broadcasts one or more arrays against one another.

    Parameters
    ----------
    x
        an arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    ret
        Each array must have the same shape. Each array must have the same dtype as its
        corresponding input array.

    """
    return _cur_framework(arrays[0]).broadcast_arrays(*arrays)


def dtype(x: Union[ivy.Array, ivy.NativeArray], as_str: bool = False) -> ivy.Dtype:
    """Get the data type for input array x.

    Parameters
    ----------
    x
        Tensor for which to get the data type.
    as_str
        Whether or not to return the dtype in string format. Default is False.

    Returns
    -------
    ret
        Data type of the array

    """
    return _cur_framework(x).dtype(x, as_str)


# noinspection PyShadowingNames


def astype(
    x: Union[ivy.Array, ivy.NativeArray], dtype: ivy.Dtype, copy: bool = True
) -> ivy.Array:
    """Copies an array to a specified data type irrespective of :ref:`type-promotion`
    rules.

    .. note::
       Casting floating-point ``NaN`` and ``infinity`` values to integral data types is
       not specified and is implementation-dependent.

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
        specifies whether to copy an array when the specified ``dtype`` matches the data
        type of the input array ``x``. If ``True``, a newly allocated array must always
        be returned. If ``False`` and the specified ``dtype`` matches the data type of
        the input array, the input array must be returned; otherwise, a newly allocated
        must be returned. Default: ``True``.

    Returns
    -------
    ret
        an array having the specified data type. The returned array must have the same
        shape as ``x``.

    Examples
    --------
    >>> x = ivy.array([1, 2])
    >>> dtype = ivy.float64
    >>> y = ivy.astype(x, dtype)
    >>> print(y)
    [1., 2.]

    """
    return _cur_framework(x).astype(x, dtype, copy)


# Extra #
# ------#

default_dtype_stack = list()
default_float_dtype_stack = list()
default_int_dtype_stack = list()


class DefaultDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, dtype):
        self._dtype = dtype

    def __enter__(self):
        set_default_dtype(self._dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_dtype()
        return self


class DefaultFloatDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, float_dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_float_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_float_dtype()
        return self


class DefaultIntDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, float_dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_int_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        return self


def dtype_bits(dtype_in: Union[ivy.Dtype, str]) -> int:
    """Get the number of bits used for representing the input data type.

    Parameters
    ----------
    dtype_in
        The data type to determine the number of bits for.

    Returns
    -------
    ret
        The number of bits used to represent the data type.

    """
    return _cur_framework(dtype_in).dtype_bits(dtype_in)


def dtype_to_str(dtype_in: Union[ivy.Dtype, str]) -> str:
    """Convert native data type to string representation.

    Parameters
    ----------
    dtype_in
        The data type to convert to string.

    Returns
    -------
    ret
        data type string 'float32'

    """
    return _cur_framework(None).dtype_to_str(dtype_in)


def dtype_from_str(dtype_in: Union[ivy.Dtype, str]) -> ivy.Dtype:
    """Convert data type string representation to native data type.

    Parameters
    ----------
    dtype_in
        The data type string to convert to native data type.

    Returns
    -------
    ret
        data type e.g. ivy.float32.

    """
    return _cur_framework(None).dtype_from_str(dtype_in)


# noinspection PyShadowingNames
def _assert_int_dtype_correct_formatting(dtype: Union[ivy.Dtype, str]):
    assert "int" in dtype


# noinspection PyShadowingNames
def _assert_float_dtype_correct_formatting(dtype: Union[ivy.Dtype, str]):
    assert "float" in dtype


# noinspection PyShadowingNames
def default_int_dtype(
    input=None, int_dtype: Union[ivy.Dtype, str] = None, as_str: bool = False
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    int_dtype

    str
         (Default value = None)
    as_str
         (Default value = False)

    Returns
    -------
        Return the input int dtype if provided, otherwise return the global default int
        dtype.

    """
    if ivy.exists(int_dtype):
        _assert_int_dtype_correct_formatting(ivy.dtype_to_str(int_dtype))
        return int_dtype
    elif ivy.exists(input):
        if ivy.is_native_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_indices_where(
                input, lambda x: x > 9223372036854775807 and x != ivy.inf
            ):
                ret = ivy.uint64
            elif ivy.nested_indices_where(
                input, lambda x: x > 2147483647 and x != ivy.inf
            ):
                ret = ivy.int64
            else:
                def_dtype = default_dtype()
                if ivy.is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.int32
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
                def_dtype = default_dtype()
                if ivy.is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.int32
    else:
        global default_int_dtype_stack
        if not default_int_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_int_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "int32"
        else:
            ret = default_int_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# len(get_binary_from_float(x)) >24 and int(get_binary_from_float(x)[24:])>0)
def _check_float64(input):
    if math.isfinite(input):
        tmp = str(input).replace("-", "").split(".")
        Exponent = int(math.floor(math.log10(abs(input)))) if input != 0 else 0
        mant = bin(int(tmp[0])).replace("0b", "")
        return (
            (input > 3.4028235 * 10**38)
            or (len(mant) > 24 and int(mant[24:]) > 0)
            or (Exponent < -126)
            or (Exponent > 127)
        )
    return False


# noinspection PyShadowingNames
def default_float_dtype(
    input=None, float_dtype: Union[ivy.Dtype, str] = None, as_str: bool = False
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    float_dtype

    str
         (Default value = None)
    as_str
         (Default value = False)

    Returns
    -------
        Return the input float dtype if provided, otherwise return the global default
        float dtype.

    """
    if ivy.exists(float_dtype):
        _assert_float_dtype_correct_formatting(ivy.dtype_to_str(float_dtype))
        return float_dtype
    elif ivy.exists(input):
        if ivy.is_native_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_indices_where(input, lambda x: _check_float64(x)):
                ret = ivy.float64
            else:
                def_dtype = default_dtype()
                if ivy.is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.float32
        elif isinstance(input, Number):
            if _check_float64(input):
                ret = ivy.float64
            else:
                def_dtype = default_dtype()
                if ivy.is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.float32
    else:
        global default_float_dtype_stack
        if not default_float_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_float_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "float32"
        else:
            ret = default_float_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# noinspection PyShadowingNames
def default_dtype(
    dtype: Union[ivy.Dtype, str] = None, item=None, as_str: bool = False
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    dtype

    str
         (Default value = None)
    item
         (Default value = None)
    as_str
         (Default value = False)

    Returns
    -------
        Return the input dtype if provided, otherwise return the global default dtype.

    """
    if ivy.exists(dtype):
        # assert the string formatting is correct if string input
        ivy.dtype_to_str(dtype)
        return dtype
    elif ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_float_dtype(item):
            return default_float_dtype(item, as_str=as_str)
        elif ivy.is_int_dtype(item):
            return default_int_dtype(item, as_str=as_str)
        elif as_str:
            return "bool"
        else:
            return dtype_from_str("bool")
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_str:
        return ivy.dtype_to_str(ret)
    return ivy.dtype_from_str(ret)


# noinspection PyShadowingNames
def set_default_dtype(dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    dtype

    """
    dtype = ivy.dtype_to_str(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


def unset_default_dtype():
    """"""
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def set_default_int_dtype(int_dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    int_dtype

    """
    int_dtype = ivy.dtype_to_str(int_dtype)
    _assert_int_dtype_correct_formatting(int_dtype)
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


def unset_default_int_dtype():
    """"""
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def set_default_float_dtype(float_dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    float_dtype

    """
    float_dtype = ivy.dtype_to_str(float_dtype)
    _assert_float_dtype_correct_formatting(float_dtype)
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


def unset_default_float_dtype():
    """"""
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


# noinspection PyShadowingBuiltins
def closest_valid_dtype(type: Union[ivy.Dtype, str, None]) -> Union[ivy.Dtype, str]:
    """Determines the closest valid datatype to the datatype passed as input.

    Parameters
    ----------
    type
        The data type for which to check the closest valid type for.

    Returns
    -------
    ret
        The closest valid data type as a native ivy.Dtype

    """
    return _cur_framework(type).closest_valid_dtype(type)


def is_int_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number]
) -> bool:
    """Determine whether the input data type is an int data-type.

    Parameters
    ----------
    dtype_in
        Datatype to test

    Returns
    -------
    ret
        Whether or not the data type is an integer data type

    """
    if ivy.is_native_array(dtype_in):
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
            if ivy.nested_indices_where(
                dtype_in, lambda x: isinstance(x, (int, np.integer))
            )
            else False
        )
    return "int" in dtype_to_str(dtype_in)


def is_float_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number]
) -> bool:
    """Determine whether the input data type is an float data-type.

    Parameters
    ----------
    dtype_in
        Datatype to test

    Returns
    -------
    ret
        Whether or not the data type is a floating point data type

    """
    if ivy.is_native_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return True if isinstance(dtype_in, (float, np.floating)) else False
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_indices_where(
                dtype_in, lambda x: isinstance(x, (float, np.floating))
            )
            else False
        )
    return "float" in dtype_to_str(dtype_in)


def result_type(
    *arrays_and_dtypes: Union[ivy.Array, ivy.NativeArray, ivy.Dtype]
) -> ivy.Dtype:
    """Returns the dtype that results from applying the type promotion rules (see
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

    """
    return _cur_framework(arrays_and_dtypes[0]).result_type(arrays_and_dtypes)


def valid_dtype(dtype_in: Union[ivy.Dtype, str, None]) -> bool:
    """Determines whether the provided data type is support by the current framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend support

    Returns
    -------
    ret
        Boolean, whether or not the data-type string is supported.

    """
    if dtype_in is None:
        return True
    return ivy.dtype_to_str(dtype_in) in ivy.valid_dtype_strs


def invalid_dtype(dtype_in: Union[ivy.Dtype, str, None]) -> bool:
    """Determines whether the provided data type is not support by the current
    framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend non-support

    Returns
    -------
    ret
        Boolean, whether the data-type string is un-supported.

    """
    if dtype_in is None:
        return False
    return ivy.dtype_to_str(dtype_in) in ivy.invalid_dtype_strs


def convert_dtype(dtype_in: Union[ivy.Dtype, str], backend: str) -> ivy.Dtype:
    """Converts a data type from one backend framework representation to another.

    Parameters
    ----------
    dtype_in
        The data-type to convert, in the specified backend representation
    backend
        The backend framework the dtype_in is represented in.

    Returns
    -------
    ret
        The data-type in the current ivy backend format

    """
    valid_backends = ["numpy", "jax", "tensorflow", "torch", "mxnet"]
    if backend not in valid_backends:
        raise Exception(
            "Invalid backend passed, must be one of {}".format(valid_backends)
        )
    ivy_backend = importlib.import_module("ivy.functional.backends.{}".format(backend))
    return ivy.dtype_from_str(ivy_backend.dtype_to_str(dtype_in))


# Function Helper #
# ----------------#

# noinspection PyShadowingNames
def _handle_dtype(dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None, arr=None):
    return ivy.dtype_from_str(ivy.default_dtype(dtype, item=arr))
