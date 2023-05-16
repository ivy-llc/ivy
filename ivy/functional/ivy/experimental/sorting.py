# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
def invert_permutation(
    x: Union[ivy.Array, ivy.NativeArray, list, tuple],
    /,
) -> ivy.Array:
    """
    Computes the inverse of an index permutation.

    Parameters
    ----------
    x
        1-D integer array-like, which represents indices of a zero-based array and is
        supposedly used to permute the array.

    Returns
    -------
    ret
        the inverse of the index permutation represented by ''x''

    Examples
    --------
    >>> a = ivy.asarray([0, 3, 1, 2])
    >>> ivy.invert_permutation(a)
    ivy.array([0, 2, 3, 1])
    """
    return ivy.current_backend().invert_permutation(x)


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
def lexsort(
    keys: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Perform an indirect stable sort with an array of keys in ascending order, with the
    last key used as primary sort order, second-to-last for secondary, and so on. Each
    row of the key must have the same length, which will also be the length of the
    returned array of integer indices, which describes the sort order.

    Parameters
    ----------
    keys
        array-like input of size (k, N).
        N is the shape of each key, key can have multiple dimension.
    axis
        axis of each key to be indirectly sorted.
        By default, sort over the last axis of each key.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array of integer(type int64) indices with shape N, that sort the keys.

    Examples
    --------
    >>> a = [1,5,1,4,3,4,4] # First column
    >>> b = [9,4,0,4,0,2,1] # Second column
    >>> ivy.lexsort([b, a]) # Sort by a, then by b
    array([2, 0, 4, 6, 5, 3, 1])
    """
    return ivy.current_backend().lexsort(keys, axis=axis, out=out)
