# global
from __future__ import annotations

from typing import Union, Tuple, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    infer_device,
    outputs_to_ivy_arrays,
    handle_nestable,
)


@outputs_to_ivy_arrays
@infer_device
@handle_nestable
@handle_exceptions
def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
) -> Tuple[ivy.Array]:
    """Returns the indices of the upper triangular part of a row by col matrix in a
    2-by-N shape (tuple of two N dimensional arrays), where the first row contains
    row coordinates of all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.  The upper triangular part
    of the matrix is defined as the elements on and above the diagonal.  The argument
    k controls which diagonal to consider. If k = 0, all elements on and above the main
    diagonal are retained. A positive value excludes just as many diagonals above the
    main diagonal, and similarly a negative value includes just as many diagonals
    below the main diagonal. The main diagonal are the set of indices
    {(i,i)} for i∈[0,min{n_rows, n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
    for examples

    Tensorflow does not support slicing 2-D tensor with tuple of tensor of indices

    Parameters
    ----------
    n_rows
       number of rows in the 2-d matrix.
    n_cols
       number of columns in the 2-d matrix. If None n_cols will be the same as n_rows
    k
       number of shifts from the main diagonal. k = 0 includes main diagonal,
       k > 0 moves upwards and k < 0 moves downwards
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an 2xN shape, tuple of two N dimensional, where first subarray (i.e. ret[0])
        contains row coordinates of all indices and the second subarray (i.e ret[1])
        contains columns indices.

    Function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = ivy.triu_indices(4,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    ivy.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    --------
    >>> x = ivy.triu_indices(4,4,1)
    >>> print(x)
    (ivy.array([0, 0, 0, 1, 1, 2]),
    ivy.array([1, 2, 3, 2, 3, 3]))

    --------
    >>> x = ivy.triu_indices(4,4,-2)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
    ivy.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))

    --------
    >>> x = ivy.triu_indices(4,2,0)
    >>> print(x)
    (ivy.array([0, 0, 1]),
    ivy.array([0, 1, 1]))

    --------
    >>> x = ivy.triu_indices(2,4,0)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1]),
    ivy.array([0, 1, 2, 3, 1, 2, 3]))

    --------
    >>> x = ivy.triu_indices(4,-4,0)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    --------
    >>> x = ivy.triu_indices(4,4,100)
    >>> print(x)
    (ivy.array([]), ivy.array([]))

    --------
    >>> x = ivy.triu_indices(2,4,-100)
    >>> print(x)
    (ivy.array([0, 0, 0, 0, 1, 1, 1, 1]), ivy.array([0, 1, 2, 3, 0, 1, 2, 3]))

    """
    return current_backend().triu_indices(n_rows, n_cols, k, device=device)
