# global
from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def difference(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray] = None,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculate the set difference of two arrays.
    Return the unique values in ``x`` that are not in ``y``.
    Parameters
    ----------
    x1 : array_like
        Input array.
    x2 : array_like
        Input comparison array.

    Returns
    -------
    difference : ndarray
        The values in ``x1`` that are not in ``x2``. This is always a flattened array.
    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/extensions/generated/signatures/setops/difference.html>`_
    in the standard.
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    Examples
    --------
    >>> x = ivy.array([1, 2, 3, 4, 5])
    >>> y = ivy.array([5, 6, 7, 8, 9])
    >>> z = ivy.difference(x, y)
    >>> print(z)
    ivy.array([1, 2, 3, 4])
    """
    return current_backend().difference(x1, x2)
