# global
from __future__ import annotations
import functools
from numbers import Number
from typing import (
    Union,
    Tuple,
    Optional,
    List,
    Sequence,
    Callable,
    Protocol,
    TypeVar,
    Iterable,
)
import numpy as np

# local
import ivy
from ivy import to_ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    infer_device,
    infer_dtype,
    handle_out_argument,
    outputs_to_ivy_arrays,
    inputs_to_native_arrays,
    inputs_to_native_shapes,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device_shifting,
)

# Helpers #
# --------#


def asarray_handle_nestable(fn: Callable) -> Callable:
    fn_name = fn.__name__

    @functools.wraps(fn)
    def _asarray_handle_nestable(*args, **kwargs):
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
        # This decorator should only be applied to ivy.asarray, so we know where
        # the container must be if there is one.
        cont_fn = getattr(ivy.Container, "static_" + fn_name)
        if isinstance(args[0], ivy.Container):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    _asarray_handle_nestable.handle_nestable = True
    return _asarray_handle_nestable


def _ivy_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
    # assumes that either all elements in a leaf list are ivy arrays
    # or none of them are
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _ivy_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and ivy.is_ivy_array(x[0]):
            x = ivy.to_native(x, nested=True)
        elif ivy.is_ivy_array(x):
            x = ivy.to_native(x)
    return x


def _shape_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _shape_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and (
            isinstance(x[0], ivy.Shape) and ivy.array_mode
        ):
            x = ivy.nested_map(x, lambda x: x.shape if isinstance(x, ivy.Shape) else x)
        elif isinstance(x, ivy.Shape) and ivy.array_mode:
            x = x.shape
    return x


def _flatten_nest(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_nest(x)
        else:
            yield x


def _remove_np_bfloat16(obj):
    # unlike other frameworks, torch and paddle do not support creating tensors
    # from numpy arrays that have bfloat16 dtype using any extension because
    # bfloat16 in not supported natively by numpy (as of version <=1.25)
    if isinstance(obj, np.ndarray) and obj.dtype.name == "bfloat16":
        return obj.tolist()
    return obj


def asarray_to_native_arrays_and_back(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back(*args, dtype=None, **kwargs):
        """
        Wrap `fn` so that input arrays are all converted to `ivy.NativeArray` instances
        and return arrays are all converted to `ivy.Array` instances.

        This wrapper is specifically for the backend implementations of
        asarray.

        It assumes either all the elements in a leaf list are ivy arrays
        or none of them are. It checks the first element of all the leaf
        list. If it is an ivy array, it converts all the elements in the
        leaf list to native otherwise it skips that leaf list.
        """
        new_arg = _ivy_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        return to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    return _asarray_to_native_arrays_and_back


def asarray_infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_dtype(*args, dtype=None, **kwargs):
        """
        Determine the correct `dtype`, and then calls the function with the `dtype`
        passed explicitly. This wrapper is specifically for the backend implementations
        of asarray.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        dtype
            The dtype for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `dtype` passed explicitly.
        """

        def _infer_dtype(obj):
            if isinstance(obj, ivy.NativeShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return ivy.default_dtype(item=obj)

        if not ivy.exists(dtype):
            arr = args[0]
            # get default dtypes for all elements
            dtype_list = [ivy.nested_map(arr, lambda x: _infer_dtype(x), shallow=False)]
            # flatten the nested structure
            dtype_list = _flatten_nest(dtype_list)
            # keep unique dtypes
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:  # handle the case of empty input
                # promote all dtypes to a single dtype
                dtype = dtype_list[0]
                # we disable precise mode to avoid wider than necessary casting
                # that might result from the mixing of int32 and float32
                with ivy.PreciseMode(False):
                    for dt in dtype_list[1:]:
                        dtype = ivy.promote_types(dtype, dt)
            else:
                dtype = ivy.default_float_dtype()
            dtype = ivy.as_native_dtype(dtype)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype.infer_dtype = True
    return _asarray_infer_dtype


def asarray_infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_device(*args, device=None, **kwargs):
        """
        Determine the correct `device`, and then calls the function with the `device`
        passed explicitly. This wrapper is specifically for the backend implementations
        of asarray.

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
        if isinstance(args[0], list):
            return fn(
                *args, device=ivy.default_device(device, as_native=True), **kwargs
            )

        # find the first array argument, if required
        arr = None if ivy.exists(device) else args[0]
        # infer the correct device
        device = ivy.default_device(device, item=arr, as_native=True)
        # call the function with device provided explicitly
        return fn(*args, device=device, **kwargs)

    _asarray_infer_device.infer_device = True
    return _asarray_infer_device


def asarray_inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        new_arg = _shape_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        return fn(*new_args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


# Type hints #
# -----------#

SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> Union[_T_co, NestedSequence[_T_co]]:
        ...

    def __len__(self, /) -> int:
        ...


# Array API Standard #
# -------------------#


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_ivy_arrays
@handle_array_function
@handle_device_shifting
@infer_device
def arange(
    start: Number,
    /,
    stop: Optional[Number] = None,
    step: Number = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return evenly spaced values within a given interval, with the spacing being
    specified.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop). For integer arguments the function
    is equivalent to the Python built-in range function, but returns an array in the
    chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    Parameters
    ----------
    start
        if stop is specified, the start of interval (inclusive); otherwise, the end of
        the interval (exclusive). If stop is not specified, the default starting value
        is 0.
    stop
        the end of the interval. Default: ``None``.
    step
        the distance between two adjacent elements (out[i+1] - out[i]). Must not be 0;
        may be negative, this results in an empty array if stop >= start. Default: 1.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from start, stop and step. If those are all integers, the output array
        dtype must be the default integer dtype; if one or more have type float, then
        the output array dtype must be the default floating-point data type. Default:
        None.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ceil((stop-start)/step) if stop - start and step have the
        same sign, and length 0 otherwise.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.arange.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> start = 5
    >>> x = ivy.arange(start)
    >>> print(x)
    ivy.array([0, 1, 2, 3, 4])

    >>> start = 1
    >>> stop = 5
    >>> x = ivy.arange(start, stop)
    >>> print(x)
    ivy.array([1, 2, 3, 4])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> x = ivy.arange(start, stop, step)
    >>> print(x)
    ivy.array([1, 3, 5, 7, 9])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> dtype = "float64"
    >>> device = "cpu"
    >>> x = ivy.arange(start, stop, step, dtype=dtype, device=device)
    >>> print(x, x.dtype, x.device)
    ivy.array([1., 3., 5., 7., 9.]) float64 cpu
    """
    return current_backend().arange(
        start, stop, step, dtype=dtype, device=device, out=out
    )


@handle_array_like_without_promotion
@handle_out_argument
@handle_array_function
@handle_device_shifting
def asarray(
    obj: Union[
        ivy.Array,
        ivy.NativeArray,
        ivy.Shape,
        ivy.NativeShape,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Convert the input to an array.

    Parameters
    ----------
    obj
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    copy
        boolean, indicating whether or not to copy the input. Default: ``None``.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array interpretation of x.

    Functional Examples
    -------------------
    With list of lists as input:
    >>> ivy.asarray([[1,2],[3,4]])
    ivy.array([[1, 2],
           [3, 4]])

    With tuple of lists as input:
    >>> ivy.asarray(([1.4,5.6,5.5],[3.1,9.1,7.5]))
    ivy.array([[1.39999998, 5.5999999 , 5.5       ],
           [3.0999999 , 9.10000038, 7.5       ]])

    With ndarray as input:
    >>> x = ivy.np.ndarray(shape=(2,2), order='C')
    >>> x
    array([[6.90786433e-310, 6.90786433e-310],
           [6.90786433e-310, 6.90786433e-310]])
    >>> ivy.asarray(x)
    ivy.array([[6.90786433e-310, 6.90786433e-310],
           [6.90786433e-310, 6.90786433e-310]])

    With :class:`ivy.Container` as input:
    >>> x = ivy.Container(a = [(1,2),(3,4),(5,6)], b = ((1,2,3),(4,5,6)))
    >>> ivy.asarray(x)
    {
        a: ivy.array([[1, 2],
                      [3, 4],
                      [5, 6]]),
        b: ivy.array([[1, 2, 3],
                      [4, 5, 6]])
    }

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.asarray.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().asarray(
        obj, copy=copy, dtype=dtype, device=device, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def zeros(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing zeros.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.NativeShape` input:
    >>> shape = (3, 5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    ivy.array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    >>> x = ivy.zeros(5)
    >>> print(x)
    ivy.array([0., 0., 0., 0., 0.])
    """
    return current_backend().zeros(shape, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def ones(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array having a specified ``shape`` and filled with ones.

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default  ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing ones.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Shape` input:

    >>> shape = (2,2)
    >>> x = ivy.ones(shape)
    >>> print(x)
    ivy.array([[1., 1.],
           [1., 1.]])

    With :class:`ivy.Dtype` input:

    >>> shape = (3,2)
    >>> d_type = object.__new__(Dtype, "int64")
    >>> y = ivy.ones(shape, dtype=d_type)
    >>> print(y)
    ivy.array([[1, 1, 1],
           [1, 1]])

    With :class:`ivy.Device` input:

    >>> shape = (3,2)
    >>> dev = object.__new__(Device, "cpu")
    >>> y = ivy.ones(shape, device=dev)
    >>> print(y)
    ivy.array([[1, 1, 1],
           [1, 1]])

    With :class:`ivy.Array` input:

    >>> shape = (1,5,2)
    >>> array = ivy.array(shape)
    >>> ivy.ones(shape, out=array)
    >>> print(array)
    ivy.array([[1.],
           [1., 1., 1., 1., 1.], [1., 1.]])
    """
    return current_backend().ones(shape, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def full_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    fill_value: Number,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array filled with ``fill_value`` and having the same ``shape`` as
    an input array ``x`` .

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    fill_value
        Scalar fill value
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------
    With :code:`int` datatype:

    >>> x = ivy.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1, 1])

    >>> fill_value = 0.000123
    >>> x = ivy.ones(5)
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With float datatype:

    >>> x = ivy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([3.0, 8.0])
    >>> fill_value = 0.000123
    >>> y = ivy.full_like(x,fill_value)
    >>> print(y)
    ivy.array([0.000123, 0.000123])

    >>> x = ivy.native_array([[3., 8., 2.], [2., 8., 3.]])
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    ivy.array([[0.000123, 0.000123, 0.000123],
           [0.000123, 0.000123, 0.000123]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1.2, 2.2324, 3.234]),
    ...                   b=ivy.array([4.123, 5.23, 6.23]))
    >>> fill_value = 15.0
    >>> y = ivy.full_like(x, fill_value)
    >>> print(y)
    {
        a: ivy.array([15., 15., 15.]),
        b: ivy.array([15., 15., 15.])
    }
    """
    return current_backend(x).full_like(
        x, fill_value, dtype=dtype, device=device, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def ones_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array filled with ones and having the same shape as an input
    array ``x``.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default  ``None``.
    device
        device on which to place the created array. If device is ``None``, the output
        array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``ones``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([1, 2, 3, 4, 5, 6])
    >>> y1 = ivy.ones_like(x1)
    >>> print(y1)
    ivy.array([1, 1, 1, 1, 1, 1])

    >>> x2 = ivy.array([[0, 1, 2],[3, 4, 5]], dtype = ivy.float32)
    >>> y2 = ivy.ones_like(x2)
    >>> print(y2)
    ivy.array([[1., 1., 1.],
            [1., 1., 1.]])

    >>> x3 = ivy.array([3., 2., 1.])
    >>> y3 = ivy.zeros(3)
    >>> ivy.ones_like(x3, out=y3)
    ivy.array([1., 1., 1.])

    With :class:`ivy.NativeArray` input:

    >>> x1 = ivy.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y1 = ivy.ones_like(x1)
    >>> print(y1)
    ivy.array([[1, 1, 1],[1, 1, 1]])


    >>> x2 = ivy.native_array([3, 8, 2, 0, 0, 2])
    >>> y2 = ivy.ones_like(x2, dtype=ivy.IntDtype('int32'), device=ivy.Device('cpu'))
    >>> print(y2)
    ivy.array([1, 1, 1, 1, 1, 1])

    # Array ``y2`` is now stored on the CPU.

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([3, 2, 1]), b=ivy.array([8, 2, 3]))
    >>> y = ivy.ones_like(x)
    >>> print(y)
    {
        a: ivy.array([1, 1, 1]),
        b: ivy.array([1, 1, 1])
    }

    Instance Method Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 8, 2, 1])
    >>> y = x.ones_like()
    >>> print(y)
    ivy.array([1, 1, 1, 1, 1])

    With :class:'ivy.Container' input:

    >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
    >>> y = x.ones_like()
    >>> print(y)
    {
        a: ivy.array([1., 1.]),
        b: ivy.array([1., 1.])
    }

    """
    return current_backend(x).ones_like(x, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def zeros_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array filled with zeros and having the same ``shape`` as an input
    array ``x``.

    Parameters
    ----------
    x
         input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``zeros``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([1, 2, 3, 4, 5, 6])
    >>> y1 = ivy.zeros_like(x1)
    >>> print(y1)
    ivy.array([0, 0, 0, 0, 0, 0])

    >>> x2 = ivy.array([[0, 1, 2],[3, 4, 5]], dtype = ivy.float32)
    >>> y2 = ivy.zeros_like(x2)
    >>> print(y2)
    ivy.array([[0., 0., 0.],
            [0., 0., 0.]])

    >>> x3 = ivy.array([3., 2., 1.])
    >>> y3 = ivy.ones(3)
    >>> ivy.zeros_like(x3, out=y3)
    ivy.array([0., 0., 0.])

    With :class:`ivy.NativeArray` input:

    >>> x1 = ivy.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y1 = ivy.zeros_like(x1)
    >>> print(y1)
    ivy.array([[0, 0, 0],[0, 0, 0]])


    >>> x2 = ivy.native_array([3, 8, 2, 0, 0, 2])
    >>> y2 = ivy.zeros_like(x2, dtype=ivy.IntDtype('int32'), device=ivy.Device('cpu'))
    >>> print(y2)
    ivy.array([0, 0, 0, 0, 0, 0])

    # Array ``y2`` is now stored on the CPU.

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([3, 2, 1]), b=ivy.array([8, 2, 3]))
    >>> y = ivy.zeros_like(x)
    >>> print(y)
    {
        a: ivy.array([0, 0, 0]),
        b: ivy.array([0, 0, 0])
    }

    Instance Method Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 8, 2, 1])
    >>> y = x.zeros_like()
    >>> print(y)
    ivy.array([0, 0, 0, 0, 0])

    With :class:'ivy.Container' input:

    >>> x = ivy.Container(a=ivy.array([3., 8.]), b=ivy.array([2., 2.]))
    >>> y = x.zeros_like()
    >>> print(y)
    {
        a: ivy.array([0., 0.]),
        b: ivy.array([0., 0.])
    }

    """
    return current_backend(x).zeros_like(x, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def tril(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the lower triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.
    k
        diagonal above which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as x. All elements above the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.tril.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).tril(x, k=k, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def triu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return the upper triangular part of a matrix (or a stack of matrices) ``x``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.    *,
    k
        diagonal below which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the upper triangular part(s). The returned array must have
        the same shape and data type as x. All elements below the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.triu.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).triu(x, k=k, out=out)


@infer_device
@infer_dtype
@handle_array_function
@outputs_to_ivy_arrays
@inputs_to_native_shapes
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_device_shifting
def empty(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an uninitialized array having a specified shape


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().empty(shape, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def empty_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return an uninitialized array with the same shape as an input array x.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from x. Deafult: ``None``.
    device
        device on which to place the created array. If device is None, the output array
        device must be inferred from x. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as x and containing uninitialized data.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).empty_like(x, dtype=dtype, device=device, out=out)


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_ivy_arrays
@handle_array_function
@infer_dtype
@infer_device
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a two-dimensional array with ones on the k diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows
        number of rows in the output array.
    n_cols
        number of columns in the output array. If None, the default number of columns in
        the output array is equal to n_rows. Default: ``None``.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and 0 to the main diagonal. Default: ``0``.
    batch_shape
        optional input that determines returning identity array shape.
        Default: ``None``.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        the device on which to place the created array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        device on which to place the created array. Default: ``None``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.eye.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances as a replacement to any of the arguments.

    Functional Examples
    -------------------

    With :'n_rows' input:

    >>> x1 = ivy.eye(3)
    >>> print(x1)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'n_cols' input:

    >>> x1 = ivy.eye(3,4)
    >>> print(x1)
    ivy.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])


    With :'k' input:

    >>> x1 = ivy.eye(3, k=1)
    >>> print(x1)
    ivy.array([[0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 0.]])


    With :'dtype' input:

    >>> x1 = ivy.eye(4, k=2, dtype=ivy.IntDtype('int32'))
    >>> print(x1)
    ivy.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])


    With :'batch_shape' input:

    >>> x1 = ivy.eye(2, 3, batch_shape=[3])
    >>> print(x1)
    ivy.array([[[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]]])
    >>> x1.shape
    (3, 2, 3)

    Suppose batch_shape = [a, b] then the returning identity
    array shape is [a, b, numRows, numColumns]


    With :'out' input:

    >>> a1 = ivy.ones(3)
    >>> ivy.eye(3, out=a1)
    >>> print(a1)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'device' input:

    >>> x1 = ivy.eye(3, device=ivy.Device('cpu'))
    >>> print(x1)
    ivy.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

    # Array ``x1`` is now stored on the CPU.
    """
    return current_backend().eye(
        n_rows,
        n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device_shifting
@infer_device
def linspace(
    start: Union[ivy.Array, ivy.NativeArray, float],
    stop: Union[ivy.Array, ivy.NativeArray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Generate a certain number of evenly-spaced values in an interval along a given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in
    an interval.

    Parameters
    ----------
    start
        First entry in the range.
    stop
        Final entry in the range.
    num
        Number of values to generate.
    axis
        Axis along which the operation is performed.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included.
    dtype
        output array data type.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.linspace.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With float input:

    >>> x = ivy.linspace(1, 2, 4)
    >>> print(x)
    ivy.array([1., 1.33333337, 1.66666663, 2.])

    >>> x = ivy.linspace(1, 2, 4, endpoint=False)
    >>> print(x)
    ivy.array([1., 1.25, 1.5 , 1.75])

    >>> x = ivy.linspace(1, 10, 4, dtype = int)
    >>> print(x)
    ivy.array([ 1,  4,  7, 10])

    >>> x = ivy.linspace(1, 2, 4, device = "gpu")
    >>> print(x)
    ivy.array([1., 1.33333337, 1.66666663, 2.])

    >>> out = ivy.array([0,0,0,0])
    >>> ivy.linspace(1, 2, 4, out = out)
    >>> print(out)
    ivy.array([1., 1.33333337, 1.66666663, 2.])

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1,2])
    >>> y = ivy.array([4,5])
    >>> z = ivy.linspace(x, y, 4, axis = 0)
    >>> print(z)
    ivy.array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])
    """
    return current_backend(start).linspace(
        start,
        stop,
        num,
        axis=axis,
        endpoint=endpoint,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def meshgrid(
    *arrays: Union[ivy.Array, ivy.NativeArray],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[ivy.Array] = None,
) -> List[ivy.Array]:
    """
    Return coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    sparse
        if True, a sparse grid is returned in order to conserve memory.
        Default: ``False``.
    indexing
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and should be ignored.
        Default: ``'xy'``.

    Returns
    -------
    ret
        list of N arrays, where ``N`` is the number of provided one-dimensional input
        arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional
        arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape
          ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape
          ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of
        length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must
        have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned
        array must have shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of
        length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned
        array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then
        each returned array must have shape ``(N, M, P)``.

        Each returned array should have the same data type as the input arrays.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
    the `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.meshgrid.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([3, 4])
    >>> xv, yv = ivy.meshgrid(x, y)
    >>> print(xv)
    ivy.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    ivy.array([[3, 3],
            [4, 4]])

    >>> x = ivy.array([1, 2, 5])
    >>> y = ivy.array([4, 1])
    >>> xv, yv = ivy.meshgrid(x, y, indexing='ij')
    >>> print(xv)
    ivy.array([[1, 1],
            [2, 2],
            [5, 5]])

    >>> print(yv)
    ivy.array([[4, 1],
            [4, 1],
            [4, 1]])

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([4, 5, 6])
    >>> xv, yv = ivy.meshgrid(x, y, sparse=True)
    >>> print(xv)
    ivy.array([[1, 2, 3]])

    >>> print(yv)
    ivy.array([[4], [5], [6]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1, 2])
    >>> y = ivy.native_array([3, 4])
    >>> xv, yv = ivy.meshgrid(x, y)
    >>> print(xv)
    ivy.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    ivy.array([[3, 3],
            [4, 4]])
    """
    return current_backend().meshgrid(
        *arrays, sparse=sparse, indexing=indexing, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_ivy_arrays
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
@infer_device
def full(
    shape: Union[ivy.Shape, ivy.NativeShape],
    fill_value: Union[float, bool],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape
        output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``fill_value``. If the fill value is an ``int``, the output
        array data type must be the default integer data type. If the fill value is a
        ``float``, the output array data type must be the default floating-point data
        type. If the fill value is a ``bool``, the output array must have boolean data
        type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array where every element is equal to `fill_value`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------

    With :class:`ivy.Shape` input:

    >>> shape = ivy.Shape((2,2))
    >>> fill_value = 8.6
    >>> x = ivy.full(shape, fill_value)
    >>> print(x)
    ivy.array([[8.6, 8.6],
               [8.6, 8.6]])

    With :class:`ivy.NativeShape` input:

    >>> shape = ivy.NativeShape((2, 2, 2))
    >>> fill_value = True
    >>> dtype = ivy.bool
    >>> device = ivy.Device('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[[True,  True],
                [True,  True]],
               [[True,  True],
                [True,  True]]])

    With :class:`ivy.NativeDevice` input:

    >>> shape = ivy.NativeShape((1, 2))
    >>> fill_value = 0.68
    >>> dtype = ivy.float64
    >>> device = ivy.NativeDevice('cpu')
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    ivy.array([[0.68, 0.68]])

    With :class:`ivy.Container` input:

    >>> shape = ivy.Container(a=ivy.NativeShape((2, 1)), b=ivy.Shape((2, 1, 2)))
    >>> fill_value = ivy.Container(a=0.99, b=False)
    >>> dtype = ivy.Container(a=ivy.float64, b=ivy.bool)
    >>> device = ivy.Container(a=ivy.NativeDevice('cpu'), b=ivy.Device('cpu'))
    >>> x = ivy.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    {
        a: ivy.array([[0.99],
                      [0.99]]),
        b: ivy.array([[[False, False]],
                      [[False, False]]])
    }


    """
    return current_backend().full(
        shape, fill_value, dtype=dtype, device=device, out=out
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def from_dlpack(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Return a new array containing the data from another (array) object with a
    ``__dlpack__`` method.

    Parameters
    ----------
    x  object
        input (array) object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.from_dlpack.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).from_dlpack(x, out=out)


# Extra #
# ------#


array = asarray


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_arrays
@handle_array_function
@handle_device_shifting
def copy_array(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    to_ivy_array: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Copy an array.

    Parameters
    ----------
    x
        array, input array containing elements to copy.
    to_ivy_array
        boolean, if True the returned array will be an ivy.Array object otherwise
        returns an ivy.NativeArray object (i.e. a torch.tensor, np.array, etc.,
        depending on the backend), defaults to True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a copy of the input array ``x``.

    Examples
    --------
    With one :class:`ivy.Array` input:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = ivy.copy_array(x)
    >>> print(y)
    ivy.array([1, 0, 1, 1])

    >>> x = ivy.array([1, 0, 1, -1])
    >>> y = ivy.zeros((1, 4))
    >>> ivy.copy_array(x, out=y)
    >>> print(y)
    ivy.array([1, 0, 1, -1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> ivy.copy_array(x, out=x)
    >>> print(x)
    ivy.array([1, 0, 1, 1])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1])
    }

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`ivy.Container` static method:

    >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),b=ivy.array([-1, 0, 1, 1, 1, 0]))
    >>> y = ivy.Container.static_copy_array(x)
    >>> print(y)
    {
        a: ivy.array([-1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`ivy.Array` instance method:

    >>> x = ivy.array([-1, 0, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([-1, 0, 1])

    >>> x = ivy.array([1, 0, 1, 1])
    >>> y = x.copy_array()
    >>> print(y)
    ivy.array([1, 0, 1, 1])

    With :class:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([1, 0, 1]),b=ivy.array([-1, 0, 1, 1]))
    >>> y = x.copy_array()
    >>> print(y)
    {
        a: ivy.array([1, 0, 1]),
        b: ivy.array([-1, 0, 1, 1])
    }
    """
    return current_backend(x).copy_array(x, to_ivy_array=to_ivy_array, out=out)


@handle_array_like_without_promotion
def native_array(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.NativeArray:
    """
    Convert the input to a native array.

    Parameters
    ----------
    x
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype
        datatype, optional. Datatype is inferred from the input data.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        A native array interpretation of x.

    Examples
    --------
    With :class:`List[Number]` input:

    >>> x = [1, 2, 3]
    >>> x_native = native_array(x)
    >>> print(x_native)
    [1. 2. 3.]

    With :class:`np.ndarray` input:
    >>> y = np.array([4, 5, 6])
    >>> y_native = native_array(y)
    >>> print(y_native)
    [4. 5. 6.]

    With :class:`ivy.Array` input:
    >>> z = ivy.array([7, 8, 9])
    >>> z_native = native_array(z)
    >>> print(z_native)
    [7. 8. 9.]
    """
    # ToDo: Make this more efficient,
    # ideally without first converting to ivy.Array with ivy.asarray and then
    # converting back to native with ivy.to_native

    return ivy.to_native(ivy.asarray(x, dtype=dtype, device=device))


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
@infer_device
def one_hot(
    indices: Union[ivy.Array, ivy.NativeArray],
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Return a one-hot array. The locations represented by indices in the parameter
    indices take value on_value, while all other locations take value off_value.

    Parameters
    ----------
    indices
        Indices for where the ones should be scattered *[batch_shape, dim]*
    depth
        Scalar defining the depth of the one-hot dimension.
    on_value
        Scalar defining the value to fill in output when indices[j] == i.
        Default: ``1``.
    off_value
        Scalar defining the value to fill in output when indices[j] != i.
        Default: ``0``.
    axis
        Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
    dtype
        The data type of the output tensor.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of zeros with the same shape and type as a, unless dtype provided which
        overrides.
    
    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([3, 1])
    >>> y = 5
    >>> z = x.one_hot(5)
    >>> print(z)
    ivy.array([[0., 0., 0., 1., 0.],
    ...    [0., 1., 0., 0., 0.]])

    >>> x = ivy.array([0])
    >>> y = 5
    >>> ivy.one_hot(x, y)
    ivy.array([[1., 0., 0., 0., 0.]])

    >>> x = ivy.array([0])
    >>> y = 5
    >>> ivy.one_hot(x, 5, out=z)
    ivy.array([[1., 0., 0., 0., 0.]])
    >>> print(z)
    ivy.array([[1., 0., 0., 0., 0.]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([1, 2]), \
        b=ivy.array([3, 1]), c=ivy.array([2, 3]))
    >>> y = 5
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: ivy.array([[0., 1., 0., 0., 0.], 
                    [0., 0., 1., 0., 0.]]),
        b: ivy.array([[0., 0., 0., 1., 0.], 
                    [0., 1., 0., 0., 0.]]),
        c: ivy.array([[0., 0., 1., 0., 0.], 
                    [0., 0., 0., 1., 0.]])
    }

    >>> x = ivy.Container(a=ivy.array([2]), \
        b=ivy.array([]), c=ivy.native_array([4]))
    >>> y = 7
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: ivy.array([[0., 0., 1., 0., 0., 0., 0.]]),
        b: ivy.array([], shape=(0, 7)),
        c: ivy.array([[0., 0., 0., 0., 1., 0., 0.]])
    }
    """
    return current_backend(indices).one_hot(
        indices,
        depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@infer_device
def logspace(
    start: Union[ivy.Array, ivy.NativeArray, float],
    stop: Union[ivy.Array, ivy.NativeArray, float],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: int = 0,
    endpoint: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Generate a certain number of evenly-spaced values in log space, in an interval along
    a given axis.

    Parameters
    ----------
    start
        First value in the range in log space. base ** start is the starting value in
        the sequence. Can be an array or a float.
    stop
        Last value in the range in log space. base ** stop is the final value in the
        sequence. Can be an array or a float.
    num
        Number of values to generate.
    base
        The base of the log space. Default is 10.0
    axis
        Axis along which the operation is performed. Relevant only if start or stop are
        array-like. Default is 0.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included. Default is
        True.
    dtype
        The data type of the output tensor. If None, the dtype of on_value is used or if
        that is None, the dtype of off_value is used, or if that is None, defaults to
        float32. Default is None.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default is
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to. Default is None.

    Returns
    -------
    ret
        Tensor of evenly-spaced values in log space.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------
    With float input:

    >>> print(ivy.logspace(1, 2, 4))
    ivy.array([ 10., 21.5443469, 46.41588834, 100.])

    >>> print(ivy.logspace(1, 2, 4, endpoint=False))
    ivy.array([10., 17.7827941, 31.6227766, 56.23413252])

    >>> print(ivy.logspace(1, 2, 4, dtype = int))
    ivy.array([10, 21, 46, 100])

    >>> out = ivy.array([0,0,0,0])
    >>> ivy.logspace(1, 2, 4, out = out)
    >>> print(out)
    ivy.array([ 10., 21.5443469, 46.41588834, 100.])

    With :class:`ivy.Array` input:
    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([4, 5])
    >>> print(ivy.logspace(x, y, 4))
    ivy.array([[1.e+01, 1.e+02],
               [1.e+02, 1.e+03],
               [1.e+03, 1.e+04],
               [1.e+04, 1.e+05])

    >>> print(ivy.logspace(x, y, 4, axis = 1))
    ivy.array([[[1.e+01, 1.e+02, 1.e+03, 1.e+04],
               [1.e+02, 1.e+03, 1.e+04, 1.e+05]]])

    >>> x = ivy.array([1, 2])
    >>> y = ivy.array([4])      # Broadcasting example
    >>> print(ivy.logspace(x, y, 4))
    ivy.array([[10., 100.]
               [100., 464.15888336]
               [1000., 2154.43469003]
               [10000., 10000.]])
    """
    result = base ** linspace(
        start,
        stop,
        num,
        endpoint=endpoint,
        axis=axis,
        dtype=dtype,
        device=device,
    )
    if ivy.exists(out):
        return ivy.inplace_update(out, result)
    return result


@handle_nestable
@outputs_to_ivy_arrays
def frombuffer(
    buffer: bytes,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> ivy.Array:
    r"""
    Interpret a buffer as a 1-dimensional array.

    .. note::
        Note that either of the following must be true:
        1. count is a positive non-zero number, and the total number of bytes
        in the buffer is equal or greater than offset plus count times the size
        (in bytes) of dtype.
        2. count is negative, and the length (number of bytes) of the buffer
        subtracted by the offset is a multiple of the size (in bytes) of dtype.

    Parameters
    ----------
    buffer
        An object that exposes the buffer interface.
    dtype
        Data-type of the returned array; default: float.
    count
        Number of items to read. -1 means all data in the buffer.
    offset
        Start reading the buffer from this offset (in bytes); default: 0.

    Returns
    -------
    out
        1-dimensional array.

    Examples
    --------
    With :class:`bytes` inputs:

    >>> x = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
    >>> y = ivy.frombuffer(x)
    >>> print(y)
    (ivy.array([1., 2.]))

    >>> x = b'\x01\x02\x03\x04'
    >>> y = ivy.frombuffer(x, dtype='int8', count=-2, offset=1)
    >>> print(y)
    (ivy.array([2, 3, 4]))

    >>> x = b'\x00<\x00@\x00B\x00D\x00E'
    >>> y = ivy.frombuffer(x, dtype='float16', count=4, offset=2)
    >>> print(y)
    (ivy.array([2., 3., 4., 5.]))
    """
    return current_backend().frombuffer(
        buffer,
        dtype=dtype,
        count=count,
        offset=offset,
    )


@handle_exceptions
@handle_nestable
@outputs_to_ivy_arrays
@infer_device
def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Tuple[ivy.Array]:
    """Return the indices of the upper triangular part of a row by col matrix in a
    2-by-N shape (tuple of two N dimensional arrays), where the first row contains
    row coordinates of all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.  The upper triangular part
    of the matrix is defined as the elements on and above the diagonal.  The argument
    k controls which diagonal to consider. If k = 0, all elements on and above the main
    diagonal are retained. A positive value excludes just as many diagonals above the
    main diagonal, and similarly a negative value includes just as many diagonals
    below the main diagonal. The main diagonal are the set of indices
    {(i,i)} for i∈[0,min{n_rows, n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
    for examples

    Tensorflow does not support slicing 2-D tensor with tuple of tensor of indices

    Parameters
    ----------
    n_rows
       number of rows in the 2-d matrix.
    n_cols
       number of columns in the 2-d matrix. If None n_cols will be the same as n_rows
    k
       number of shifts from the main diagonal. k = 0 includes main diagonal,
       k > 0 moves upwards and k < 0 moves downwards
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an 2xN shape, tuple of two N dimensional, where first subarray (i.e. ret[0])
        contains row coordinates of all indices and the second subarray (i.e ret[1])
        contains columns indices.

    Function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.triu_indices(4,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    ivy.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,1)
    >>> print(x)
    (ivy.array([0, 0, 0, 1, 1, 2]),
    ivy.array([1, 2, 3, 2, 3, 3]))

    >>> x = ivy.triu_indices(4,4,-2)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
    ivy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,2,0)
    >>> print(x)
    (ivy.array([0, 0, 1]),
    ivy.array([0, 1, 1]))

    >>> x = ivy.triu_indices(2,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1]),
    ivy.array([0, 1, 2, 3, 1, 2, 3]))

    >>> x = ivy.triu_indices(4,-4,0)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(4,4,100)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    >>> x = ivy.triu_indices(2,4,-100)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1]), ivy.array([0, 1, 2, 3, 0, 1, 2, 3]))

    """
    return current_backend().triu_indices(n_rows, n_cols, k, device=device)
