# global
from typing import Optional, Union

# local
import ivy
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_nestable,
    handle_out_argument,
    to_native_arrays_and_back,
)

# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def msort(
    a: Union[ivy.Array, ivy.NativeArray, list, tuple],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Return a copy of an array sorted along the first axis.

    Parameters
    ----------
    a
        array-like input.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        sorted array of the same type and shape as a

    Examples
    --------
    >>> a = ivy.randint(10, size=(2,3))
    >>> ivy.msort(a)
    ivy.array(
        [[6, 2, 6],
         [8, 9, 6]]
        )
    """
    return ivy.current_backend().msort(a, out=out)
