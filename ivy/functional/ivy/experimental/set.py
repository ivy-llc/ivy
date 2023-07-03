# global
from typing import Union, Tuple

# local
import ivy
from ivy.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
def intersection(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
    """
    Find the intersection of two arrays.

    Parameters
    ----------
    x1
        Input array. Will be flattened if not already 1D.
    x2
        Input array. Will be flattened if not already 1D.

    assume_unique
        Bool. If True, the input arrays are both assumed to be unique,
        which can speed up the calculation.
        If True but x1 or x2 are not unique, incorrect results and
        out-of-bounds indices could result.
        Default is False.

    return_indices
        Bool. If True, the indices which correspond
        to the intersection of the two arrays are returned.
        The first instance of a value is used if there are multiple.
        Default is False.

    Returns
    -------
    ret
        Tuple.
        - Sorted 1D Array of common and unique elements.
        - The indices of the first occurrences of the common values in x1.
        Only provided if return_indices is True.
        - The indices of the first occurrences of the common values in x2.
        Only provided if return_indices is True.

    Examples
    --------
    With :class:`ivy.Array` inputs:

    >>> x = ivy.array([5., 2., 4., 9.])
    >>> y = ivy.array([2., 1., 5., 2.])
    >>> z = ivy.intersection(x, y)
    >>> print(z)
    ivy.array([2., 5.])

    >>> x = ivy.array([[5, 2, 9], [0, 1, 3]])
    >>> y = ivy.array([[2, 0, 4], [0, 3, 1]])
    >>> z = ivy.intersection(x, y)
    >>> print(z)
    ivy.array([0, 1, 2, 3])

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([5, 1, 7]), b=ivy.array([1, 0, 9]))
    >>> y = ivy.Container(a=ivy.array([4, 7, 4]), b=ivy.array([3, 0, 9]))
    >>> z = ivy.intersection(x,y)
    >>> print(z)
    {
        a: ivy.array([7]),
        b: ivy.array([0, 9])
    }

    With a combination of :class:`ivy.Array`
    and :class:`ivy.Container` inputs:

    >>> x = ivy.array([9., 0., 1.])
    >>> y = ivy.Container(a=ivy.array([2., 1.]), b=ivy.array([1., 0.]))
    >>> z = ivy.intersection(x, y)
    >>> print(z)
    {
        a: ivy.array([1.]),
        b: ivy.array([0., 1.])
    }

    >>> x = ivy.array([[1, 2, 9], [0, 3, 1]])
    >>> y = ivy.array([[1, 3, 9], [3, 2, 4]])
    >>> z = ivy.intersection(x, y)
    >>> print(z)
    ivy.array([1, 2, 3, 9])
    """
    return ivy.current_backend(x1, x2).intersection(
        x1, x2, assume_unique=assume_unique, return_indices=return_indices
    )
