# global
from typing  import Union, Optional, Tuple, Literal


# local
import ivy
inf = float('inf')



def vector_norm(x: Union[ivy.Array, ivy.NativeArray], 
                p: Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                    -> ivy.Array:

                    
    """
    Compute the vector p-norm.

    :param x: Input array.
    :type x: array
    :param p: Order of the norm. Default is 2.
    :type p: int or str, optional
    :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms.
                 Default is None, in which case the flattened array is considered.
    :type axis: int or sequence of ints, optional
    :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with
                     size one. With this option the result will broadcast correctly against the original x.
                     Default is False.
    :type keepdims: bool, optional
    :return: Vector norm of the array at specified axes.
    """
    if p == -float('inf'):
        return ivy.reduce_min(ivy.abs(x), axis, keepdims)
    elif p == float('inf'):
        return ivy.reduce_max(ivy.abs(x), axis, keepdims)
    elif p == 0:
        return ivy.reduce_sum(ivy.cast(x != 0, 'float32'), axis, keepdims)
    x_raised = x ** p
    return ivy.reduce_sum(x_raised, axis, keepdims) ** (1/p)