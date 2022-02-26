# global
from typing import Union, Optional, Tuple, Literal

# local
import ivy
inf = float('inf')


# noinspection PyShadowingBuiltins
def vector_norm(x: Union[ivy.Array, ivy.NativeArray],
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False,
                ord: Union[int, float, Literal[inf, -inf]] = 2)\
        -> ivy.Array:

    """
    Computes the vector norm of a vector (or batch of vectors) x.

    :param x: input array. Should have a floating-point data type.
    :param axis: If an integer, ``axis`` specifies the axis (dimension) along which to compute vector norms. If an
                 n-tuple, ``axis`` specifies the axes (dimensions) along which to compute batched vector norms. If
                 ``None``, the vector norm must be computed over all array values (i.e., equivalent to computing the
                 vector norm of a flattened array). Negative indices must be supported. Default: ``None``.
    :param keepdims: If ``True``, the axes (dimensions) specified by ``axis`` must be included in the result as
                     singleton dimensions, and, accordingly, the result must be compatible with the input array.
                     Otherwise, if ``False``, the axes (dimensions) specified by ``axis`` must not be included in the
                     result. Default: ``False``.
    :param keepdims: If True, the axes (dimensions) specified by axis must be included in the result as singleton
                     dimensions, and, accordingly, the result must be compatible with the input array (see
                     Broadcasting). Otherwise, if False, the axes (dimensions) specified by axis must not be included
                     in the result. Default: False.
    :param ord: order of the norm. The following mathematical norms must be supported:
        +------------------+----------------------------+
        | ord              | description                |
        +==================+============================+
        | 1                | L1-norm (Manhattan)        |
        +------------------+----------------------------+
        | 2                | L2-norm (Euclidean)        |
        +------------------+----------------------------+
        | inf              | infinity norm              |
        +------------------+----------------------------+
        | (int,float >= 1) | p-norm                     |
        +------------------+----------------------------+
        The following non-mathematical "norms" must be supported:
        +------------------+--------------------------------+
        | ord              | description                    |
        +==================+================================+
        | 0                | sum(a != 0)                    |
        +------------------+--------------------------------+
        | -1               | 1./sum(1./abs(a))              |
        +------------------+--------------------------------+
        | -2               | 1./sqrt(sum(1./abs(a)\*\*2))   |
        +------------------+--------------------------------+
        | -inf             | min(abs(a))                    |
        +------------------+--------------------------------+
        | (int,float < 1)  | sum(abs(a)\*\*ord)\*\*(1./ord) |
        +------------------+--------------------------------+
        Default: ``2``.
    :return: an array containing the vector norms. If ``axis`` is ``None``, the returned array must be a
             zero-dimensional array containing a vector norm. If ``axis`` is a scalar value (``int`` or ``float``), the
             returned array must have a rank which is one less than the rank of ``x``. If ``axis`` is a ``n``-tuple, the
             returned array must have a rank which is ``n`` less than the rank of ``x``. The returned array must have a
             floating-point data type determined by type-promotion.
    """
    if ord == -float('inf'):
        return ivy.reduce_min(ivy.abs(x), axis, keepdims)
    elif ord == float('inf'):
        return ivy.reduce_max(ivy.abs(x), axis, keepdims)
    elif ord == 0:
        return ivy.reduce_sum(ivy.cast(x != 0, 'float32'), axis, keepdims)
    x_raised = x ** ord
    return ivy.reduce_sum(x_raised, axis, keepdims) ** (1/ord)
