# global
import math
import numpy as np
from numbers import Number
from typing import Union, Tuple, List, Optional, Callable, Iterable

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_arrays,
    handle_nestable,
)


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


# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def astype(
    x: Union[ivy.Array, ivy.NativeArray],
    dtype: Union[ivy.Dtype, ivy.NativeDtype],
    /,
    *,
    copy: bool = True,
    out: Optional[ivy.Array] = None,
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
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the specified data type. The returned array must have the same
        shape as ``x``.

    Examples
    --------
    >>> x = ivy.array([1, 2])
    >>> y = ivy.astype(x, dtype = ivy.float64)
    >>> print(y)
    ivy.array([1., 2.])
    """
    return current_backend(x).astype(x, dtype, copy=copy, out=out)


@to_native_arrays_and_back
@handle_nestable
def broadcast_arrays(*arrays: Union[ivy.Array, ivy.NativeArray]) -> List[ivy.Array]:
    """Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays
        an arbitrary number of arrays to-be broadcasted.
        Each array must have the same shape. Each array must have the same dtype as its
        corresponding input array.

    Returns
    -------
    ret
        A list containing broadcasted arrays of type `ivy.Array`

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x1 = ivy.array([1, 2, 3])
    >>> x2 = ivy.array([4, 5, 6])
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [ivy.array([1, 2, 3]), ivy.array([4, 5, 6])]

    With :code:`ivy.NativeArray` inputs:

    >>> x1 = ivy.native_array([0.3, 4.3])
    >>> x2 = ivy.native_array([3.1, 5])
    >>> x3 = ivy.native_array([2, 0])
    >>> y = ivy.broadcast_arrays(x1, x2, x3)
    [ivy.array([0.3, 4.3]), ivy.array([3.1, 5.]), ivy.array([2, 0])]

    With mixed :code:`ivy.Array` and :code:`ivy.NativeArray` inputs:

    >>> x1 = ivy.array([1, 2])
    >>> x2 = ivy.native_array([0.3, 4.3])
    >>> y = ivy.broadcast_arrays(x1, x2)
    >>> print(y)
    [ivy.array([1, 2]), ivy.array([0.3, 4.3])]

    With :code:`ivy.Container` inputs:

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

    With mixed :code:`ivy.Array` and :code:`ivy.Container` inputs:

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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
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
    With :code:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.broadcast_to(x, (3, 3))
    >>> print(y)
    ivy.array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]])

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0.1 , 0.3])
    >>> y = ivy.broadcast_to(x, (3, 2))
    >>> print(y)
    ivy.array([[0.1, 0.3],
               [0.1, 0.3],
               [0.1, 0.3]])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
        b=ivy.array([4, 5, 6]))
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
    return current_backend(x).broadcast_to(x, shape)


@inputs_to_native_arrays
@handle_nestable
def can_cast(
    from_: Union[ivy.Dtype, ivy.Array, ivy.NativeArray],
    to: ivy.Dtype,
    /,
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

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.data_type_functions.can_cast.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
     With :code:`ivy.Dtype` input:

    >>> print(ivy.can_cast(ivy.uint8, ivy.int32))
    True

    >>> print(ivy.can_cast(ivy.float64, 'int64'))
    False

    With :code:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> print(ivy.can_cast(x, ivy.float64))
    True

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[-1, -1, -1],\
                              [1, 1, 1]],\
                            dtype='int16')
    >>> print(ivy.can_cast(x, 'uint8'))
    False

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),\
                          b=ivy.array([3, 4, 5]))
    >>> print(ivy.can_cast(x, 'int64'))
    {
        a: false,
        b: true
    }
    """
    return current_backend(from_).can_cast(from_, to)


@inputs_to_native_arrays
@handle_nestable
def finfo(
    type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray],
    /,
) -> Finfo:
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

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.data_type_functions.can_cast.html>`_ # noqa
    in the standard.

    Examples
    --------
    With :code:`ivy.Dtype` input:

    >>> ivy.finfo(ivy.float32)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :code:`str` input:

    >>> ivy.finfo('float32')
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :code:`ivy.Array` input:

    >>> x = ivy.array([1.3,2.1,3.4], dtype=ivy.float64)
    >>> ivy.finfo(x)
    finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
    max=1.7976931348623157e+308, dtype=float64)

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([0.7,8.4,3.14], dtype=ivy.float16)
    >>> ivy.finfo(x)
    finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)

    With :code:`ivy.Container` input:

    >>> c = ivy.Container(x=ivy.array([-9.5,1.8,-8.9], dtype=ivy.float16), /
                          y=ivy.array([7.6,8.1,1.6], dtype=ivy.float64))
    >>> ivy.finfo(c)
    {
        x: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16),
        y: finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
           max=1.7976931348623157e+308, dtype=float64)
    }

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([0.7,8.4,3.14], dtype=ivy.float32)
    >>> x.finfo()
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    Using :code:`ivy.Container` instance method:

    >>> c = ivy.Container(x=ivy.array([1.2,3.5,8.], dtype=ivy.float64), /
                          y=ivy.array([1.3,2.1,3.4], dtype=ivy.float16))
    >>> c.finfo()
    {
        x: finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
                 max=1.7976931348623157e+308, dtype=float64)
        y: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16),
    }

    """
    return current_backend(None).finfo(type)


@inputs_to_native_arrays
@handle_nestable
def iinfo(
    type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray],
    /,
) -> Iinfo:
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
    
    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.data_type_functions.iinfo.html>`_ # noqa
    in the standard.

    Examples
    --------
    With :code:`ivy.Dtype` input:

    >>> ivy.iinfo(ivy.int32)
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    >>> ivy.iinfo(ivy.uint64)
    iinfo(min=0, max=18446744073709551615, dtype=uint64)

    With :code:`str` input:

    >>> ivy.iinfo('int32')
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    >>> ivy.iinfo('int64')
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

    With :code:`ivy.Array` input:

    >>> x = ivy.array([13,21,34]) 
    >>> ivy.iinfo(x)
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    >>> x = ivy.array([13,21,34], dtype=ivy.int8)
    >>> ivy.iinfo(x)
    iinfo(min=-128, max=127, dtype=int8)
    
    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([7,84,314], dtype=ivy.int16)
    >>> ivy.iinfo(x)
    iinfo(min=-32768, max=32767, dtype=int16)
    
    >>> x = ivy.native_array([7,84,314], dtype=ivy.int64)
    >>> ivy.iinfo(x)
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

    With :code:`ivy.Container` input:

    >>> c = ivy.Container(x=ivy.array([-9,1800,89], dtype=ivy.int16), \
                          y=ivy.array([76,-81,16], dtype=ivy.int32))
    >>> ivy.iinfo(c)
    {
        x: iinfo(min=-32768, max=32767, dtype=int16),
        y: iinfo(min=-2147483648, max=2147483647, dtype=int32)
    }

    >>> c = ivy.Container(x=ivy.array([0,1800,89], dtype=ivy.uint16), \
                          y=ivy.array([76,81,16], dtype=ivy.uint32))
    >>> ivy.iinfo(c)
    {
        x: iinfo(min=0, max=65535, dtype=uint16),
        y: iinfo(min=0, max=4294967295, dtype=uint32)
    }

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([109,8400,14], dtype=ivy.int32)
    >>> x.iinfo()
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    >>> x = ivy.array([-119,122,14], dtype=ivy.int8))
    >>> x.iinfo()
    iinfo(min=-128, max=127, dtype=int8)

    Using :code:`ivy.Container` instance method:

    >>> c = ivy.Container(x=ivy.array([-9,1800,89], dtype=ivy.int16), \
                          y=ivy.array([76,-81,16], dtype=ivy.int32))
    >>> c.iinfo()
    {
        x: iinfo(min=-32768, max=32767, dtype=int16),
        y: iinfo(min=-2147483648, max=2147483647, dtype=int32)
    }

    """
    return current_backend(None).iinfo(type)


@inputs_to_native_arrays
@handle_nestable
def result_type(
    *arrays_and_dtypes: Union[ivy.Array, ivy.NativeArray, ivy.Dtype]
) -> ivy.Dtype:
    """
    Returns the dtype that results from applying the type promotion rules (see
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.data_type_functions.result_type.html>`_ # noqa
    in the standard.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([3, 4, 5])
    >>> y = ivy.array([3., 4., 5.])
    >>> print(ivy.result_type(x, y))
    float64

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([3., 4, 7.5])
    >>> y = ivy.native_array([3, 4, 7])
    >>> print(ivy.result_type(x, y))
    float64

    With :code:`ivy.Dtype` input:

    >>> print(ivy.result_type(ivy.uint8, ivy.uint64))
    uint64

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.array([3, 4, 5]))
    >>> print(x.a.dtype)
    int32

    >>> print(ivy.result_type(x, ivy.float64))
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


class DefaultDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, dtype: ivy.Dtype):
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
    def __init__(self, float_dtype: ivy.Dtype):
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
    def __init__(self, float_dtype: ivy.Dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_int_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        return self


class DefaultUintDtype:
    """"""

    def __init__(self, uint_dtype: ivy.UintDtype):
        self._uint_dtype = uint_dtype

    def __enter__(self):
        set_default_uint_dtype(self._uint_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_uint_dtype()
        return self


def dtype_bits(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str], /) -> int:
    """Get the number of bits used for representing the input data type.

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
    With :code:`ivy.Dtype` inputs:

    >>> x = ivy.dtype_bits(ivy.float32)
    >>> print(x)
    32

    >>> x = ivy.dtype_bits('int64')
    >>> print(x)
    64

    >>> x = ivy.dtype_bits(ivy.uint16)
    >>> print(x)
    16

    With :code:`ivy.NativeDtype` inputs:

    >>> x = ivy.dtype_bits(ivy.native_int8)
    >>> print(x)
    8

    >>> x = ivy.dtype_bits(ivy.native_bool)
    >>> print(x)
    1
    """
    return current_backend(dtype_in).dtype_bits(dtype_in)


def as_ivy_dtype(dtype_in: Union[ivy.Dtype, str], /) -> ivy.Dtype:
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
    return current_backend(None).as_ivy_dtype(dtype_in)


def as_native_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype], /) -> ivy.NativeDtype:
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
    return current_backend(None).as_native_dtype(dtype_in)


# len(get_binary_from_float(x)) >24 and int(get_binary_from_float(x)[24:])>0)
# noinspection PyShadowingBuiltins
def _check_float64(input) -> bool:
    if math.isfinite(input):
        tmp = str(input).replace("-", "").split(".")
        exponent = int(math.floor(math.log10(abs(input)))) if input != 0 else 0
        mant = bin(int(tmp[0])).replace("0b", "")
        return (
            (input > 3.4028235 * 10**38)
            or (len(mant) > 24 and int(mant[24:]) > 0)
            or (exponent < -126)
            or (exponent > 127)
        )
    return False


# noinspection PyShadowingBuiltins
def closest_valid_dtype(type: Union[ivy.Dtype, str, None], /) -> Union[ivy.Dtype, str]:
    """Determines the closest valid datatype to the datatype passed as input.

    Parameters
    ----------
    type
        The data type for which to check the closest valid type for.

    Returns
    -------
    ret
        The closest valid data type as a native ivy.Dtype

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.add.html>`_ # noqa
    in the standard.

    Examples
    --------
    With :code:`ivy.Dtype` input:

    >>> xType = ivy.float16
    >>> yType = ivy.closest_valid_dtype(xType)
    >>> print(yType)
    float16

    >>> xType = ivy.int8
    >>> yType = ivy.closest_valid_dtype(xType)
    >>> print(yType)
    int8

    With :code:`ivy.Native_dtype` inputs:

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


# noinspection PyShadowingNames,PyShadowingBuiltins
@handle_nestable
def default_float_dtype(
    *,
    input=None,
    float_dtype: Optional[Union[ivy.FloatDtype, ivy.NativeDtype]] = None,
    as_native: Optional[bool] = None,
) -> Union[ivy.Dtype, str, ivy.NativeDtype]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    float_dtype

    as_native
         (Default value = None)

    Returns
    -------
        Return the input float dtype if provided, otherwise return the global default
        float dtype.

    """
    if ivy.exists(float_dtype):
        if as_native is True:
            return ivy.as_native_dtype(float_dtype)
        elif as_native is False:
            return ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
        return float_dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
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
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.FloatDtype(ivy.as_ivy_dtype(ret))


# noinspection PyShadowingNames
def default_dtype(
    *, dtype: Union[ivy.Dtype, str] = None, item=None, as_native: Optional[bool] = None
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    dtype

    item
         (Default value = None)
    as_native
         (Default value = None)

    Returns
    -------
        Return the input dtype if provided, otherwise return the global default dtype.

    """
    if ivy.exists(dtype):
        if as_native is True:
            return ivy.as_native_dtype(dtype)
        elif as_native is False:
            return ivy.as_ivy_dtype(dtype)
        return dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_float_dtype(item):
            return ivy.default_float_dtype(input=item, as_native=as_native)
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


# noinspection PyShadowingNames,PyShadowingBuiltins
def default_int_dtype(
    *,
    input=None,
    int_dtype: Optional[Union[ivy.IntDtype, ivy.NativeDtype]] = None,
    as_native: Optional[bool] = False,
) -> Union[ivy.IntDtype, ivy.NativeDtype]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    int_dtype

    as_native
         (Default value = None)

    Returns
    -------
        Return the input int dtype if provided, otherwise return the global default int
        dtype.

    """
    if ivy.exists(int_dtype):
        if as_native is True:
            return ivy.as_native_dtype(int_dtype)
        elif as_native is False:
            return ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
        return int_dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
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
                def_dtype = ivy.default_dtype()
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
                def_dtype = ivy.default_dtype()
                if ivy.is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.int32
    else:
        global default_int_dtype_stack
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


def default_uint_dtype(
    *,
    input=None,
    uint_dtype: Optional[Union[ivy.UintDtype, ivy.NativeDtype]] = None,
    as_native: Optional[bool] = None,
) -> Union[ivy.UintDtype, ivy.NativeDtype]:
    """Returns the default uint dtype currently set. If input number or array is
    given, returns uint dtype according to input, else uint32 by default.

    Parameters
    ----------
    input
        Number or array for inferring default uint dtype. Optional.
    uint_dtype
        Uint dtype to be returned as defualt. Optional.
    as_native
        Whether to return the default uint dtype as native dtype. Optional.

    Returns
    -------
        Return the input uint dtype if provided, otherwise return the global default
        uint dtype.

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
            if ivy.nested_indices_where(
                input, lambda x: x > 4294967295 and x != ivy.inf
            ):
                ret = ivy.uint64
            else:
                def_dtype = ivy.default_dtype()
                if ivy.is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.uint32
        elif isinstance(input, Number):
            if input > 4294967295 and input != ivy.inf and ivy.backend != "torch":
                ret = ivy.uint64
            else:
                def_dtype = ivy.default_dtype()
                if ivy.is_uint_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.uint32
    else:
        global default_uint_dtype_stack
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


@handle_nestable
def dtype(
    x: Union[ivy.Array, ivy.NativeArray], *, as_native: bool = False
) -> Union[ivy.Dtype, ivy.NativeDtype]:
    """Get the data type for input array x.

    Parameters
    ----------
    x
        Tensor for which to get the data type.
    as_native
        Whether or not to return the dtype in string format. Default is False.

    Returns
    -------
    ret
        Data type of the array

    Functional Method Examples
    --------------------------

    With :code:`ivy.Array` inputs:

    >>> x1 = ivy.array([1, 0, 1, -1, 0])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    int32

    >>> x1 = ivy.array([1.0, 2.0, 3.5, 4.5, 5, 6])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    float32

    With :code:`ivy.Native_Array` inputs:

    >>> x1 = ivy.native_array([1, 0, 1, -1, 0])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    int32

    >>> x1 = ivy.native_array([1.0, 2.0, 3.5, 4.5, 5, 6])
    >>> y = ivy.dtype(x1)
    >>> print(y)
    float32

    With :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1, 0, -1, 0, 1]), \
                    b=ivy.array([1, 0, -1, 0, 1]))
    >>> y = ivy.dtype(x.a)
    >>> print(y)
    int32

    >>> x = ivy.Container(a=ivy.native_array([1.0, 2.0, -1.0, 4.0, 1.0]), \
                            b=ivy.native_array([1, 0, 0, 0, 1]))
    >>> y = ivy.dtype(x.a)
    >>> print(y)
    float32

    Instance Method Examples
    ------------------------

    With :code:`ivy.Array` inputs:

    >>> x = ivy.array([1, 2, 3])
    >>> y = x.dtype
    >>> print(y)
    int32

    With :code:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                      b=ivy.array([2, 3, 4]))
    >>> y = x.dtype()
    >>> print(y)
    {
        a: int32,
        b: int32
    }

    """
    return current_backend(x).dtype(x, as_native)


@handle_nestable
def function_supported_dtypes(fn: Callable) -> Tuple:
    """Returns the supported data types of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the supported dtype attribute

    Returns
    -------
    ret
        The supported data types of the function

    Examples
    --------
    >>> ivy.set_backend('torch')
    >>> print(ivy.function_supported_dtypes(ivy.acosh))
    ()
    """
    if not _is_valid_dtypes_attributes(fn):
        raise Exception(
            "supported_dtypes and unsupported_dtypes attributes cannot both \
             exist in a particular backend"
        )
    supported_dtypes = tuple()
    if hasattr(fn, "supported_dtypes"):
        fn_supported_dtypes = fn.supported_dtypes
        if isinstance(fn_supported_dtypes, dict):
            backend_str = ivy.current_backend_str()
            if backend_str in fn_supported_dtypes:
                supported_dtypes += fn_supported_dtypes[backend_str]
            if "all" in fn_supported_dtypes:
                supported_dtypes += fn_supported_dtypes["all"]
        else:
            supported_dtypes += fn_supported_dtypes
    return tuple(set(supported_dtypes))


@handle_nestable
def function_unsupported_dtypes(fn: Callable) -> Tuple:
    """Returns the unsupported data types of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the unsupported dtype attribute

    Returns
    -------
    ret
        The unsupported data types of the function

    Examples
    --------
    >>> ivy.set_backend('torch')
    >>> print(ivy.function_unsupported_dtypes(ivy.acosh))
    ('float16','uint16','uint32','uint64')

    """
    if not _is_valid_dtypes_attributes(fn):
        raise Exception(
            "supported_dtypes and unsupported_dtypes attributes cannot both \
             exist in a particular backend"
        )
    unsupported_dtypes = ivy.invalid_dtypes
    if hasattr(fn, "unsupported_dtypes"):
        fn_unsupported_dtypes = fn.unsupported_dtypes
        if isinstance(fn_unsupported_dtypes, dict):
            backend_str = ivy.current_backend_str()
            if backend_str in fn_unsupported_dtypes:
                unsupported_dtypes += fn_unsupported_dtypes[backend_str]
            if "all" in fn_unsupported_dtypes:
                unsupported_dtypes += fn_unsupported_dtypes["all"]
        else:
            unsupported_dtypes += fn_unsupported_dtypes
    return tuple(set(unsupported_dtypes))


def invalid_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str, None], /) -> bool:
    """
    Determines whether the provided data type is not support by
    the current framework.

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
    with :code:`ivy.Dtype` inputs:

    >>> print(ivy.invalid_dtype(None))
    False

    >>> print(ivy.invalid_dtype('uint64'))
    False

    >>> print(ivy.invalid_dtype(ivy.float64))
    False

    >>> print(ivy.invalid_dtype('float32'))
    False



    with :code:`ivy.NativeDtype` inputs:

    >>> print(ivy.invalid_dtype(ivy.native_uint8))
    False

    >>> print(ivy.invalid_dtype(ivy.native_float32))
    False

    >>> print(ivy.invalid_dtype('native_bool'))
    True

    >>> print(ivy.invalid_dtype('native_float64'))
    True

    >>> print(ivy.invalid_dtype(ivy.native_int16))
    False

    >>> print(ivy.invalid_dtype('native_int32'))
    True

    >>> print(ivy.invalid_dtype(ivy.native_float16))
    False

    >>> print(ivy.invalid_dtype(ivy.native_int64))
    False

    >>> print(ivy.invalid_dtype(ivy.native_int8))
    False

    >>> print(ivy.invalid_dtype('native_uint64'))
    True


    """
    if dtype_in is None:
        return False
    return ivy.as_ivy_dtype(dtype_in) in ivy.invalid_dtypes


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
    :code:`ivy.Container` instances in place of any of the arguments.

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
            if ivy.nested_indices_where(
                dtype_in,
                lambda x: isinstance(x, (bool, np.bool)) and not type(x) == int,
            )
            else False
        )
    return "bool" in ivy.as_ivy_dtype(dtype_in)


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
    :code:`ivy.Container` instances in place of any of the arguments.

    Examples
    --------
     With :code:`ivy.Dtype` input:

    >>> x = ivy.is_int_dtype(ivy.int8)
    >>> print(x)
    True

    >>> x = ivy.is_int_dtype(ivy.int32)
    >>> print(x)
    True

    >>> x = ivy.is_int_dtype(ivy.float64)
    >>> print(x)
    False

    >>> x = ivy.is_int_dtype(ivy.bool)
    >>> print(x)
    False



    With :code:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> print(x.dtype)
    float32

    >>> print(ivy.is_int_dtype(x))
    False

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[-1, -1, -1], [1, 1, 1]], \
        dtype = ivy.int16)
    >>> print(x.dtype)
    torch.int16

    >>> print(ivy.is_int_dtype(x))
    True

    With :code:`Number` input:

    >>> x = 1
    >>> print(ivy.is_int_dtype(x))
    True

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
        b=ivy.array([3, 4, 5]))
    >>> print(x.a.dtype, x.b.dtype)
    float32 int32

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
            if ivy.nested_indices_where(
                dtype_in,
                lambda x: isinstance(x, (int, np.integer)) and not type(x) == bool,
            )
            else False
        )
    return "int" in ivy.as_ivy_dtype(dtype_in)


def check_float(x):
    """
    Helper function to check if the input is a float or a float-like object.

    Parameters
    ----------
    x : any
        Input to check.

    Returns
    -------
    ret : bool
        "True" if the input is a float or a float-like object, otherwise "False".
    """
    return isinstance(x, (int, np.float)) and not type(x) == bool


@inputs_to_native_arrays
@handle_nestable
def is_float_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    *,
    out: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number] = None,
) -> bool:
    """
    Determine whether the input data type is a float dtype.

    Parameters
    ----------
    dtype_in : Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number]
        The array or data type to check

    Returns
    -------
    ret : bool
        Whether or not the array or data type is of a floating point dtype

    Examples
    --------
    With :code:`ivy.Dtype` input:

    >>> x = ivy.is_float_dtype(ivy.float32)
    >>> print(x)
    True

    >>> x = ivy.is_float_dtype(ivy.int64)
    >>> print(x)
    False

    >>> x = ivy.is_float_dtype(ivy.int32)
    >>> print(x)
    False

    >>> x = ivy.is_float_dtype(ivy.bool)
    >>> print(x)
    False

    >>> arr = ivy.array([1.2, 3.2, 4.3], dtype=ivy.float32)
    >>> print(arr.is_float_dtype())
    True

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3, 4, 5]))
    >>> print(x.a.dtype, x.b.dtype)
    float32 int32
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
            if ivy.nested_indices_where(
                dtype_in, lambda x: isinstance(x, (float, np.floating))
            )
            else False
        )
    return "float" in as_ivy_dtype(dtype_in)


@inputs_to_native_arrays
@handle_nestable
def is_uint_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is a uint dtype.

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
        return ivy.nested_indices_where(
            dtype_in, lambda x: isinstance(x, np.unsignedinteger)
        )
    return "uint" in as_ivy_dtype(dtype_in)


def promote_types(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
) -> ivy.Dtype:
    """
    Promotes the datatypes type1 and type2, returning the data type they promote to

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote

    Returns
    -------
    ret
        The type that both input types promote to
    """
    try:
        ret = ivy.promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]
    except KeyError:
        raise Exception("these dtypes are not type promotable")
    return ret


def set_default_dtype(dtype: Union[ivy.Dtype, ivy.NativeDtype, str], /):
    """
    Sets the datatype dtype as default data type

    Parameters
    ----------
    dtype
        the data_type to set as default data type

    Examples
    --------
    With :code:`ivy.Dtype` input:

    >>> ivy.set_default_dtype("float64")
    >>> ivy.default_dtype_stack
        ['float64']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype(ivy.bool)
    >>> ivy.default_dtype_stack
        ['bool']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype(ivy.int32)
    >>> ivy.default_dtype_stack
        ['int32']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype('uint8')
    >>> ivy.default_dtype_stack
        ['uint8']
    >>> ivy.unset_default_dtype()

    With :code:`ivy.NativeDtype` input:

    >>> ivy.set_default_dtype(ivy.native_int32)
    >>> ivy.default_dtype_stack
        ['int32']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype('native_bool')
    >>> ivy.default_dtype_stack
        ['native_bool']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype(ivy.native_uint64)
    >>> ivy.default_dtype_stack
        ['uint64']
    >>> ivy.unset_default_dtype()

    >>> ivy.set_default_dtype('native_float64')
    >>> ivy.default_dtype_stack
        ['native_float64']
    >>> ivy.unset_default_dtype()
    """
    dtype = ivy.as_ivy_dtype(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


def set_default_float_dtype(float_dtype: Union[ivy.Dtype, str], /):
    """Summary.

    Parameters
    ----------
    float_dtype

    """
    float_dtype = ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


def set_default_int_dtype(int_dtype: Union[ivy.Dtype, str], /):
    """Summary.

    Parameters
    ----------
    int_dtype

    """
    int_dtype = ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


def set_default_uint_dtype(uint_dtype: Union[ivy.Dtype, str], /):
    """Set the uint dtype to be default.

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
    global default_uint_dtype_stack
    default_uint_dtype_stack.append(uint_dtype)


def type_promote_arrays(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
) -> Tuple:
    """
    Type promote the input arrays, returning new arrays with the shared correct
    data type

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


def unset_default_dtype():
    """"""
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def unset_default_float_dtype():
    """"""
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def unset_default_int_dtype():
    """"""
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)


def unset_default_uint_dtype():
    """Reset the current default uint dtype to the previous state

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


def valid_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str, None], /) -> bool:
    """Determines whether the provided data type is supported by the
    current framework.

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
    with :code:`ivy.Dtype` inputs:

    >>> print(ivy.valid_dtype(None))
    True

    >>> print(ivy.valid_dtype('float16'))
    True

    >>> print(ivy.valid_dtype('float32'))
    True

    >>> print(ivy.valid_dtype(ivy.float64))
    True

    >>> print(ivy.valid_dtype('bool'))
    True

    >>> print(ivy.valid_dtype(ivy.int8))
    True

    >>> print(ivy.valid_dtype(ivy.int64))
    True

    >>> print(ivy.valid_dtype(ivy.uint8))
    True

    with :code:`ivy.NativeDtype` inputs:

    >>> print(ivy.valid_dtype('native_bool'))
    False

    >>> print(ivy.valid_dtype(ivy.native_float16))
    True

    >>> print(ivy.valid_dtype(ivy.native_float32))
    True

    >>> print(ivy.valid_dtype('native_float64'))
    False

    >>> print(ivy.valid_dtype(ivy.native_int8))
    True

    >>> print(ivy.valid_dtype(ivy.native_int16))
    True

    >>> print(ivy.valid_dtype('native_int32'))
    False

    >>> print(ivy.valid_dtype(ivy.native_int64))
    True

    >>> print(ivy.valid_dtype(ivy.native_uint8))
    True

    >>> print(ivy.valid_dtype('native_uint64'))
    False
    """
    if dtype_in is None:
        return True
    return ivy.as_ivy_dtype(dtype_in) in ivy.valid_dtypes


def promote_types_of_inputs(
    x1: Union[ivy.NativeArray, Number, Iterable[Number]],
    x2: Union[ivy.NativeArray, Number, Iterable[Number]],
    /,
) -> Tuple[ivy.NativeArray, ivy.NativeArray]:
    """
    Promotes the dtype of the given native array inputs to a common dtype
    based on type promotion rules. While passing float or integer values or any
    other non-array input to this function, it should be noted that the return will
    be an array-like object. Therefore, outputs from this function should be used
    as inputs only for those functions that expect an array-like or tensor-like objects,
    otherwise it might give unexpected results.
    """
    try:
        if (hasattr(x1, "dtype") and hasattr(x2, "dtype")) or (
            not hasattr(x1, "dtype") and not hasattr(x2, "dtype")
        ):
            x1 = ivy.asarray(x1)
            x2 = ivy.asarray(x2)
            promoted = promote_types(x1.dtype, x2.dtype)
            x1 = ivy.asarray(x1, dtype=promoted)
            x2 = ivy.asarray(x2, dtype=promoted)
        else:
            if hasattr(x1, "dtype"):
                x1 = ivy.asarray(x1)
                x2 = ivy.asarray(x2, dtype=x1.dtype)
            else:
                x1 = ivy.asarray(x1, dtype=x2.dtype)
                x2 = ivy.asarray(x2)
        x1, x2 = ivy.to_native(x1), ivy.to_native(x2)
        return x1, x2
    except Exception:
        raise
