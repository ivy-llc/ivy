from typing import Optional, Union, Tuple
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_device,
    handle_backend_invalid,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def unravel_index(
    indices: Union[ivy.Array, ivy.NativeArray],
    shape: Tuple[int],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> Tuple[ivy.Array]:
    """Convert a flat index or array of flat indices into a tuple of coordinate
    arrays.

    Parameters
    ----------
    indices
        Input array.
    shape
        The shape of the array to use for unraveling indices.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Tuple with arrays of type int32 that have the same shape as the indices array.

    Examples
    --------
    >>> indices = ivy.array([22, 41, 37])
    >>> ivy.unravel_index(indices, (7,6))
    (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
    """
    return ivy.current_backend(indices).unravel_index(indices, shape, out=out)
