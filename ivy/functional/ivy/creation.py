# global
from __future__ import annotations
import functools
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence, Callable, Protocol, TypeVar
import numpy as np

# local
import ivy
from ivy import to_ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    infer_device,
    infer_dtype,
    handle_out_argument,
    outputs_to_ivy_arrays,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like,
)

# Helpers #
# --------#


def asarray_handle_nestable(fn: Callable) -> Callable:
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
        # This decorator should only be applied to ivy.asarray, so we know where
        # the container must be if there is one.
        cont_fn = getattr(ivy.Container, "static_" + fn_name)
        if isinstance(args[0], ivy.Container):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an ivy or a native array.
        return fn(*args, **kwargs)

    new_fn.handle_nestable = True
    return new_fn


def _ivy_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
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


def asarray_to_native_arrays_and_back(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, dtype=None, **kwargs):
        """
        Wraps `fn` so that input arrays are all converted to `ivy.NativeArray` instances
        and return arrays are all converted to `ivy.Array` instances. This wrapper is
        specifically for the backend implementations of asarray.
        """
        # When possible we want to not nest this
        # because nested calls introduce massive overhead
        # and the checks to see if we can avoid it are cheap
        new_arg = _ivy_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        return to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    return new_fn


def asarray_infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, device=None, **kwargs):
        """
        Determines the correct `device`, and then calls the function with the `device`
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
            return fn(*args, device=ivy.default_device(as_native=True), **kwargs)

        # find the first array argument, if required
        arr = None if ivy.exists(device) else args[0]
        # infer the correct device
        device = ivy.default_device(device, item=arr, as_native=True)
        # call the function with device provided explicitly
        return fn(*args, device=device, **kwargs)

    new_fn.infer_device = True
    return new_fn


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


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
@handle_nestable
@handle_exceptions
def arange(
    start: Number,
    stop: Optional[Number] = None,
    step: Optional[Number] = 1,
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns evenly spaced values within a given interval, with the spacing being
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.arange.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend().arange(
        start, stop, step, dtype=dtype, device=device, out=out
    )


@handle_out_argument
@handle_exceptions
def asarray(
    obj: Union[
        ivy.Array,
        ivy.NativeArray,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Converts the input to an array.

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


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.asarray.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    return current_backend().asarray(
        obj, copy=copy, dtype=dtype, device=device, out=out
    )


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
def zeros(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with zeros.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.zeros.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> shape = (3, 5)
    >>> x = ivy.zeros(shape)
    >>> print(x)
    ivy.array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    """
    return current_backend().zeros(shape, dtype=dtype, device=device, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
def ones(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with ones.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.ones.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> shape = (2,2)
    >>> y = ivy.ones(shape)
    >>> print(y)
    ivy.array([[1., 1.],
           [1., 1.]])

    """
    return current_backend().ones(shape, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@infer_dtype
@handle_nestable
@handle_exceptions
@handle_array_like
def full_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    fill_value: Number,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array filled with ``fill_value`` and having the same ``shape`` as
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.full_like.html>`_ # noqa
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

    >>> x = ivy.Container(a=ivy.array([1.2,2.2324,3.234]),b=ivy.array([4.123,5.23,6.23]))
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


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
def ones_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array filled with ones and having the same shape as an input
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.ones_like.html>`_ # noqa
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


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
def zeros_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array filled with zeros and having the same ``shape`` as an input
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.zeros_like.html>`_ # noqa
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def tril(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.tril.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).tril(x, k=k, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def triu(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.triu.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).triu(x, k=k, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
def empty(
    shape: Union[ivy.Shape, ivy.NativeShape],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a new array of given shape and type, filled with zeros.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.empty.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend().empty(shape, dtype=dtype, device=device, out=out)


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
def empty_like(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an uninitialized array with the same shape as an input array x.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.empty_like.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).empty_like(x, dtype=dtype, device=device, out=out)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
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
    """Returns a two-dimensional array with ones on the k diagonal and zeros elsewhere.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.eye.html>`_ # noqa
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
    # Suppose batch_shape = [a, b] then the returning identity array shape is [a, b, numRows, numColumns]


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


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
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
    """Generates a certain number of evenly-spaced values in an interval along a given
    axis.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.linspace.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

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


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
def meshgrid(
    *arrays: Union[ivy.Array, ivy.NativeArray],
    sparse: bool = False,
    indexing: str = "xy",
) -> List[ivy.Array]:
    """Returns coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    sparse
        if True, a sparse grid is returned in order to conserve memory. Default: ``False``.
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
    the `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.meshgrid.html>`_ # noqa
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
        ivy.array([[4],
                [5],
                [6]])

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
    return current_backend().meshgrid(*arrays, sparse=sparse, indexing=indexing)


@outputs_to_ivy_arrays
@handle_out_argument
@infer_device
@handle_nestable
@handle_exceptions
def full(
    shape: Union[ivy.Shape, ivy.NativeShape],
    fill_value: Union[float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns a new array having a specified ``shape`` and filled with ``fill_value``.

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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.full.html>`_ # noqa
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def from_dlpack(
    x: Union[ivy.Array, ivy.NativeArray], /, *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Returns a new array containing the data from another (array) object with a
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
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.from_dlpack.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    """
    return current_backend(x).from_dlpack(x, out=out)


# Extra #
# ------#


array = asarray


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def copy_array(
    x: Union[ivy.Array, ivy.NativeArray], *, out: Optional[ivy.Array] = None
) -> ivy.Array:
    """Copy an array.

    Parameters
    ----------
    x
        array, input array containing elements to copy.
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
    return current_backend(x).copy_array(x, out=out)


@handle_exceptions
def native_array(
    x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> ivy.NativeArray:
    """Converts the input to a native array.

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

    """
    # ToDo: Make this more efficient,
    # ideally without first converting to ivy.Array with ivy.asarray and then
    # converting back to native with ivy.to_native

    return ivy.to_native(ivy.asarray(x, dtype=dtype, device=device))


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
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
    """Returns a one-hot array. The locations represented by indices in the parameter
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


@to_native_arrays_and_back
@handle_out_argument
@infer_dtype
@infer_device
@handle_nestable
@handle_exceptions
@handle_array_like
def logspace(
    start: Union[ivy.Array, ivy.NativeArray, int],
    stop: Union[ivy.Array, ivy.NativeArray, int],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: Optional[int] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    device: Union[ivy.Device, ivy.NativeDevice] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Generates a certain number of evenly-spaced values in log space, in an interval
    along a given axis.

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
    base
        The base of the log space. Default is 10.0
    axis
        Axis along which the operation is performed.
    dtype
        The data type of the output tensor. If None, the dtype of on_value is used or if
        that is None, the dtype of off_value is used, or if that is None, defaults to
        float32.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Functional Examples
    -------------------
    With General input:

    >>> x = ivy.logspace(2.0,3.0,num=4)
    >>> print(x)
    ivy.array([100., 215.443469, 464.15888336, 1000.])

    >>> x = ivy.logspace(2.0, 3.0, num=4,axis=0,base=3)
    >>> print(x)
    ivy.array([9., 12.98024613, 18.72075441, 27.])

    With :class:`ivy.Array` input:

    >>> y1 = ivy.array([[1, 2, 3], [7, 4, 5]])
    >>> y2 = ivy.array([[1, 3, 9], [5, 7, 11]])
    >>> op = ivy.logspace(y1, y2, 2, axis=2,base=2)
    >>> print(op)
    ivy.array([[[2.000e+00, 2.000e+00]
                [4.000e+00, 8.000e+00]
                [8.000e+00, 5.120e+02]],

               [[1.280e+02, 3.200e+01]
                [1.600e+01, 1.280e+02]
                [3.200e+01, 2.048e+03]]])

    >>> x1 = ivy.array([1, 2, 3, 4, 5, 6])
    >>> x2 = ivy.array([7, 11, 2, 9, 7, 6])
    >>> op = ivy.logspace(x1,x2,2,axis=1)
    >>> print(op)
    ivy.array([[1.e+01, 1.e+07]
               [1.e+02, 1.e+11]
               [1.e+03, 1.e+02]
               [1.e+04, 1.e+09]
               [1.e+05, 1.e+07]
               [1.e+06, 1.e+06]])

    >>> x = ivy.array([3., 2., 1.])
    >>> y = ivy.array([13., 72., 11.])
    >>> op = ivy.logspace(x, y, 3, axis=1,dev_str=ivy.Device('cpu'))
    >>> print(op)
    ivy.array([[1.e+03, 1.e+08, 1.e+13]
               [1.e+02, 1.e+37, 1.e+72]
               [1.e+01, 1.e+06, 1.e+11]])

    # Array ``op`` is now stored on the CPU.

    >>> y1 = ivy.array([[1, 3, 9], [5, 7, 11]])
    >>> y2 = ivy.zeros(3)
    >>> op = ivy.logspace(y1, y2, 3, axis=1,base=1)
    >>> print(op)
    ivy.array([[[1., 1., 1.]
                [1., 1., 1.]
                [1., 1., 1.]],

               [[1., 1., 1.]
                [1., 1., 1.]
                [1., 1., 1.]]])

    >>> x1 = ivy.array([[5, 13, 9], [15, 6, 7]])
    >>> x2 = ivy.ones(3)
    >>> op = ivy.logspace(x1, x2, 2, axis=0, base=2)
    >>> print(op)
    ivy.array([[[3.2000e+01, 8.1920e+03, 5.1200e+02]
                [3.2768e+04, 6.4000e+01, 1.2800e+02]],
               [[2.0000e+00 2.0000e+00 2.0000e+00]
                [2.0000e+00 2.0000e+00 2.0000e+00]]])

     With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([5., 6.]), b=ivy.array([21., 12.]))
    >>> op = ivy.logspace(x.a,x.b,num=3)
    >>> print(op)
    ivy.array([[1.e+05, 1.e+13, 1.e+21]
               [1.e+06, 1.e+09, 1.e+12]])

    >>> x = ivy.Container(a=ivy.array([.3, .8]), b=ivy.array([.2, .2]))
    >>> y = ivy.Container(a=ivy.array([.3, .2]), b=ivy.array([.7, .9]))
    >>> op1 = ivy.logspace(x.a,y.b,num=3)
    >>> op2 = ivy.logspace(x.b,y.a,num=3)
    >>> op = ivy.logspace(op1,op2,num=5,base=2)
    >>> print(op)
    ivy.array([[[3.98688589, 3.71321804, 3.45833531, 3.22094823, 2.99985588]
                [8.95241962, 7.04342969, 5.54150764, 4.35985143, 3.43016841]
                [32.26442336, 19.12940149, 11.34171831, 6.72444323, 3.98688589]],

               [[79.31783778, 34.97866642, 15.42537138, 6.80249153, 2.99985588]
                [135.24747445, 52.19416699, 20.14256517, 7.77333857, 2.99985588]
                [246.1309628, 81.78050455, 27.17273296, 9.02852606, 2.99985588]]])

    """
    return current_backend(start).logspace(
        start, stop, num, base=base, axis=axis, dtype=dtype, device=device, out=out
    )
