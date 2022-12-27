from typing import Optional, Union
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_nestable
@handle_exceptions
def isin(
    elements: Union[ivy.Array, ivy.NativeArray],
    test_elements: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> ivy.Array:
    """Tests if each element of elements is in test_elements.

    Parameters
    ----------
    elements
        input array
    test_elements
        values against which to test for each input element
    assume_unique
        If True, assumes both elements and test_elements contain unique elements,
        which can speed up the calculation. Default value is False.
    invert
        If True, inverts the boolean return array, resulting in True values for
        elements not in test_elements. Default value is False.

    Returns
    -------
    ret
        output a boolean array of the same shape as elements that is True for elements
        in test_elements and False otherwise.

    Examples
    --------
    >>> x = ivy.array([[10, 7, 4], [3, 2, 1]])
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.isin(x, y)
    ivy.array([[False, False, False], [ True,  True,  True]])

    >>> x = ivy.array([3, 2, 1, 0])
    >>> y = ivy.array([1, 2, 3])
    >>> ivy.isin(x, y, invert=True)
    ivy.array([False, False, False,  True])
    """
    return ivy.current_backend().isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert
    )
