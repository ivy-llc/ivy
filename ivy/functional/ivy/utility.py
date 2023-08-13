# global
from typing import Union, Optional, Sequence

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    handle_backend_invalid,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def all(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Test whether all input array elements evaluate to ``True`` along a specified
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
    axis.
        axis or axes along which to perform a logical AND reduction. By default, a
        logical AND reduction must be performed over the entire array. If a tuple of
        integers, logical AND reductions must be performed over multiple axes. A valid
        ``axis`` must be an integer on the interval ``[-N, N]``, where ``N`` is the rank
        (number of dimensions) of ``x``. If an ``axis`` is specified as a negative
        integer, the function must determine the axis along which to perform a reduction
        by counting backward from the last dimension (where ``-1`` refers to the last
        dimension). If provided an invalid ``axis``, the function must raise an
        exception. Default ``None``.
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


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.all.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicit
    y,but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.all(x)
    >>> print(y)
    ivy.array(True)

    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.zeros((1,1), dtype='bool')
    >>> a = ivy.all(x, axis=0, out = y, keepdims=True)
    >>> print(a)
    ivy.array([[False]])

    >>> x = ivy.array(False)
    >>> y = ivy.all(ivy.array([[0, 4],[1, 5]]), axis=(0,1), out=x, keepdims=False)
    >>> print(y)
    ivy.array(False)

    >>> x = ivy.array(False)
    >>> y = ivy.all(ivy.array([[[0], [1]], [[1], [1]]]), axis=(0,1,2), out=x,
    ...             keepdims=False)
    >>> print(y)
    ivy.array(False)

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([3, 4, 5]))
    >>> y = ivy.all(x)
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }

    >>> x = ivy.Container(a=ivy.native_array([0, 1, 2]),b=ivy.array([3, 4, 5]))
    >>> y = ivy.all(x)
    >>> print(y)
    {
        a: ivy.array(False),
        b: ivy.array(True)
    }
    """
    return ivy.current_backend(x).all(x, axis=axis, keepdims=keepdims, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def any(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Test whether any input array element evaluates to True along a specified axis.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input array. The array on which the logical OR operation is performed.

    axis : Optional[Union[int, Sequence[int]]], optional
        Axis or axes along which to perform a logical OR reduction. By default,
        a logical OR reduction is performed over the entire array. If a tuple of
        integers, logical OR reductions are performed over multiple axes. A valid
        axis must be an integer on the interval [-N, N], where N is the rank
        (number of dimensions) of x. If an axis is specified as a negative integer,
        the function determines the axis along which to perform a reduction by counting
        backward from the last dimension (where -1 refers to the last dimension). If
        provided an invalid axis, the function raises an IvyIndexError. Default: None.

    keepdims : bool, optional
        If True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions,and, accordingly, the result must be compatible with the
        input array (see Broadcasting).Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result.Default: False.

    out : Optional[ivy.Array], optional
        Output array, for writing the result to. It must have a shape that the inputs
        broadcast to.Default: None.

    Returns
    -------
    ret : ivy.Array
        If a logical OR reduction was performed over the entire array, the returned
        array must be a zero-dimensional array containing the test result; otherwise,
        the returned array must be a non-zero-dimensional array containing the test
        results. The returned array must have a data type of bool.

    Raises
    ------
    IvyIndexError
        If the provided axis is invalid.

    Notes
    -----
    - Positive infinity, negative infinity, and NaN must evaluate to True.
      Explanation: The logical OR operation considers positive infinity, negative
      infinity, and NaN as True.

    - If x is an empty array or the size of the axis (dimension) along which to
      evaluate elements is zero,the test result must be False.
      Explanation: If the input array x is empty or the axis along which the elements
      are evaluated has a size of zero,the result of the logical OR operation must be
      False.

    This method conforms to the Array API Standard.
    Both the description and the type hints above assume an array input for simplicity,
    but this function is nestable, and therefore also accepts Ivy Container instances
    in place of the arguments.

    Examples
    --------
    Example 1:
    >>> import ivy
    >>> x = ivy.array([[True, False], [False, False]], dtype='bool')
    >>> result = any(x, axis=0)
    >>> print(result)
    [True, False]
        Explanation: The function performs a logical OR reduction along axis 0, which
        results in [True, False].

    Example 2:
    >>> x = ivy.array([[1, 0], [0, 0]], dtype='int32')
    >>> result = any(x, axis=0)
    >>> print(result)
    [True, False]
        Explanation: The input array is converted to a boolean array before applying
        the logical OR operation.

    Example 3:
    >>> x = ivy.Container(a=ivy.array([True, False]),b= ivy.array([False, False]))
    >>> result = any(x, axis=0)
    >>> print(result)
    [True, False]
        Explanation: The function accepts Ivy Container instances as input, performing
        the logical OR operation along axis 0.

    Example 4:
    >>> result_flattened = any(x, axis=None)
    >>> print(result_flattened)
    True
        Explanation: If axis is set to None, the function performs a logical OR
        reduction over the entire array and returns a single boolean value.

    Example 5:
    >>> x = ivy.array([False, False])
    >>> output_array = ivy.array([False, True], dtype='bool')
    >>> result_out = any(x, axis=0, out=output_array)
    >>> print(result_out)
    [True, False]
        Explanation: The function supports the out argument for inplace updates.

    Example 6:
    >>> x = ivy.array([[0, 3], [1, 4]])
    >>> result_axis_seq = any(x, axis=(0, 1))
    >>> print(result_axis_seq)
    True
        Explanation: The axis argument accepts both an integer and a sequence of
        integers.
    """
    return ivy.current_backend(x).any(x, axis=axis, keepdims=keepdims, out=out)


# Extra #
# ----- #


def save(item, filepath, format=None):
    if isinstance(item, ivy.Container):
        if format is not None:
            item.cont_save(filepath, format=format)
        else:
            item.cont_save(filepath)
    elif isinstance(item, ivy.Module):
        item.save(filepath)
    else:
        raise ivy.utils.exceptions.IvyException("Unsupported item type for saving.")


@staticmethod
def load(filepath, format=None, type="module"):
    if type == "module":
        return ivy.Module.load(filepath)
    elif type == "container":
        if format is not None:
            return ivy.Container.cont_load(filepath, format=format)
        else:
            return ivy.Container.cont_load(filepath)
    else:
        raise ivy.utils.exceptions.IvyException("Unsupported item type for loading.")
