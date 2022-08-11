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
    '''
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of a, 
        containing the indices of the non-zero elements in that dimension. 
        The values in a are always tested and returned in row-major, C-style order.

    To group the indices by element, rather than dimension, 
        use `argwhere`, which returns a row for each non-zero element.

    `numpy.nonzero` has the same behavior with `ivy.nonzero`
    '''
    return ivy.nonzero(a)
