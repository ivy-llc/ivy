# global
from typing import Union, Optional, Sequence, Any

# local
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def all(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Tests whether all input array elements evaluate to ``True`` along a specified
    axis.

    .. note::
       Positive infinity, negative infinity, and NaN must evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis (dimension) along which to
       evaluate elements is zero, the test result must be ``True``.

    Parameters
    ----------
    x
        input array.
    axis
        axis or axes along which to perform a logical AND reduction. By default, a
        logical AND reduction must be performed over the entire array. If a tuple of
        integers, logical AND reductions must be performed over multiple axes. A valid
        ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank
        (number of dimensions) of ``x``. If an ``axis`` is specified as a negative
        integer, the function must determine the axis along which to perform a reduction
        by counting backward from the last dimension (where ``-1`` refers to the last
        dimension). If provided an invalid ``axis``, the function must raise an
        exception. Default  ``None``.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        if a logical AND reduction was performed over the entire array, the returned
        array must be a zero-dimensional array containing the test result; otherwise,
        the returned array must be a non-zero-dimensional array containing the test
        results. The returned array must have a data type of ``bool``.

    Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.all(x)
    >>> print(y)
    ivy.array(True)

    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.zeros((1,1), dtype='bool')
    >>> a = ivy.all(x, axis=0, out = y, keepdims=True)
    >>> print(a)
    ivy.array([[False]])

    >>> x=ivy.array(False)
    >>> y=ivy.all(ivy.array([[0, 4],[1, 5]]), axis=(0,1), out=x, keepdims=False)
    >>> print(y)
    ivy.array(False)

    >>> x=ivy.array(False)
    >>> y=ivy.all(ivy.array([[[0],[1]],[[1],[1]]]), \
    axis=(0,1,2), out=x, keepdims=False)
    >>> print(y)
    ivy.array(False)

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([1, 2, 3])
    >>> y = ivy.all(x)
    >>> print(y)
    ivy.array(True)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = ivy.all(x)
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }

    >>> x = ivy.Container(a=ivy.native_array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = ivy.all(x)
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([1, 2, 3])
    >>> y = x.all()
    >>> print(y)
    ivy.array(True)

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = x.all()
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }

    >>> x = ivy.Container(a=ivy.native_array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = x.all()
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }

    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/sig
    natures.utility_functions.all.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicit
    y,but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.
    """
    return ivy.current_backend(x).all(x, axis=axis, keepdims=keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def any(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Tests whether any input array element evaluates to ``True`` along a specified
    axis.

    .. note::
       Positive infinity, negative infinity, and NaN must evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis (dimension) along which to
       evaluate elements is zero, the test result must be ``False``.

    Parameters
    ----------
    x
        input array.
    axis
        axis or axes along which to perform a logical OR reduction. By default, a
        logical OR reduction must be performed over the entire array. If a tuple of
        integers, logical OR reductions must be performed over multiple axes. A valid
        ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank
        (number of dimensions) of ``x``. If an ``axis`` is specified as a negative
        integer, the function must determine the axis along which to perform a reduction
        by counting backward from the last dimension (where ``-1`` refers to the last
        dimension). If provided an invalid ``axis``, the function must raise an
        exception. Default: ``None``.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        if a logical OR reduction was performed over the entire array, the returned
        array must be a zero-dimensional array containing the test result; otherwise,
        the returned array must be a non-zero-dimensional array containing the test
        results. The returned array must have a data type of ``bool``.

        Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 4])
    >>> y = ivy.any(x)
    >>> print(y)
    ivy.array(True)

    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.zeros((1,1), dtype='bool')
    >>> a = ivy.any(x, axis=0, out = y, keepdims=True)
    >>> print(a)
    ivy.array([[True]])

    >>> x=ivy.array(False)
    >>> y=ivy.any(ivy.array([[0, 3],[1, 4]]), axis=(0,1), out=x, keepdims=False)
    >>> print(y)
    ivy.array(True)

    >>> x=ivy.array(False)
    >>> y=ivy.any(ivy.array([[[0],[1]],[[1],[1]]]), \
    axis=(0,1,2), out=x, keepdims=False)
    >>> print(y)
    ivy.array(True)

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([2, 3, 4])
    >>> y = ivy.any(x)
    >>> print(y)
    ivy.array(True)

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = ivy.any(x)
    >>> print(y)
    {
        a: ivy.array(True),
        b: ivy.array(True)
    }

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = x.any()
    >>> print(y)
    {
        a: ivy.array(True),
        b: ivy.array(True)
    }

    Instance Method Examples
    ------------------------

    Using :code:`ivy.Array` instance method:

    >>> x = ivy.array([2, 3, 4])
    >>> y = x.any()
    >>> print(y)
    ivy.array(True)

    Using :code:`ivy.Container` instance method:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = x.any()
    >>> print(y)
    {
        a: ivy.array(True),
        b: ivy.array(True)
    }

    >>> x = ivy.Container(a=ivy.native_array([0, 1, 2]), \
                          b=ivy.array([3, 4, 5]))
    >>> y = x.any()
    >>> print(y)
    {
        a: ivy.array(True),
        b: ivy.array(True)
    }
    """
    return ivy.current_backend(x).any(x, axis=axis, keepdims=keepdims, out=out)


# Extra #
# ------#
@to_native_arrays_and_back
@handle_nestable
def is_tensor(x: Any)  -> bool:
    """Tests whether the object x is a tensor

    Parameters
    ----------
    x
        Any input object. It can be String, Array etc

    Returns
    -------
    ret
        Returns ``bool`` datatype with True or False value

        Functional Examples
    -------------------

    With :code:`ivy.Array` input:

    >>> x = ivy.array([2, 3, 4])
    >>> y = ivy.is_tensor(x)
    >>> print(y)
    False
    
    ivy.set_backend('torch')
    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.is_tensor(x)
    >>> print(y)
    True

    ivy.set_backend('tensorflow')
    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.is_tensor(x)
    >>> print(y)
    True

    ivy.set_backend('jax')
    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.is_tensor(x)
    >>> print(y)
    False

    ivy.set_backend('mxnet')
    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.is_tensor(x)
    >>> print(y)
    False
    """
    return ivy.current_backend(x).is_tensor(x)