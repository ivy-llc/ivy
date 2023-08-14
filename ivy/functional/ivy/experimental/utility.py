# global
from typing import Optional

# local
import ivy
from ivy import handle_out_argument, handle_nestable
from ivy.utils.exceptions import handle_exceptions


@handle_out_argument
@handle_nestable
@handle_exceptions
def optional_get_element(
    x: Optional[ivy.Array] = None,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    If the input is a tensor or sequence type, it returns the input.
    If the input is an optional type, it outputs the element in the input.
    It is an error if the input is an empty optional-type (i.e. does not have an element)
    and the behavior is undefined in this case.

    Parameters
    ----------
    x
        Input array
    out
        Optional output array, for writing the result to.

    Returns
    -------
    ret
        Input array if it is not None

    """
    if x is None:
        raise ivy.utils.exceptions.IvyError("The requested optional input has no value.")
    return x
