# local
import ivy


def where(cond, x1=None, x2=None, /):
    if x1 is None and x2 is None:
        # numpy where behaves as np.asarray(condition).nonzero() when x and y
        # not included
        return ivy.asarray(cond).nonzero()
    elif x1 is not None and x2 is not None:
        return ivy.where(cond, x1, x2)
    else:
        raise TypeError("where takes either 1 or 3 arguments")


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of a,
        containing the indices of the non-zero elements in that dimension.
        The values in a are always tested and returned in row-major, C-style order.

    To group the indices by element, rather than dimension,
        use `argwhere`, which returns a row for each non-zero element.

    `numpy.nonzero` has the same behavior with `ivy.nonzero`
    """
    return ivy.nonzero(a)


def argmin(x, /, *, axis=None, keepdims=False, out=None):
    """
    Returns the indices of the minimum values along a specified axis. When the
    minimum value occurs multiple times, only the indices corresponding to the first
    occurrence are returned.
    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        minimum value of the flattened array. Default = None.
    keepdims
        if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default = False.
    out
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the minimum value; otherwise, a non-zero-dimensional array
        containing the indices of the minimum values. The returned array must have the
        default array index data type.
    Returns
    -------
    ret
        Array containing the indices of the minimum values across the specified axis.
    """
    return ivy.argmin(x, axis=axis, out=out, keepdims=keepdims)
