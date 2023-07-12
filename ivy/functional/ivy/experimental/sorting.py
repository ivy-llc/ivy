# global
from typing import Union, Optional

# local
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_device_shifting,
)
from ivy.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_nestable
@to_native_arrays_and_back
@handle_device_shifting
def invert_permutation(
    x: Union[ivy.Array, ivy.NativeArray, list, tuple],
    /,
) -> ivy.Array:
    """
    Compute the inverse of an index permutation.

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
@handle_device_shifting
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

@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def argpartition(
    a: Union[ivy.Array, ivy.NativeArray],
    kth,
    /,
    *,
    axis: int = -1,
    kind: Literal["introselect"] = 'introselect',
    order: Optional[List[str]] = None,
) -> ivy.Array:
    
    """
     Perform an indirect partition along the given axis using the algorithm specified by 
     the kind keyword. It returns an array of indices of the same shape as `a` that 
     index data along the given axis in partitioned order.

    Parameters
    ----------
    a : array-like object

        Array to sort. 

    kth : integer or sequence of integers   

        Element index to partition by. The k-th element will be in its final sorted position and all smaller elements will 
        be moved before it and all larger elements behind it.
        
    
    axis : int, optional 

        Axis along which to sort. The default is -1 (the last axis). If None, `a` is flattened before sorting.


    kind : {'introselect'}, optional  

        Selection algorithm. Default is 'introselect'.


    order : str or list of str, optional  

        When `a` is an array with fields defined, this argument specifies which fields to compare first, second, etc.
        

    Returns

    -------

    index_array : ndarray[int], same shape as `a`.
    
            Array of indices that partition `a` along the specified axes.


    This function conforms to the `Array API Standard <https://data-apis.org/array-api/latest/>`_.
    This docstring is an extension of the standard's documentation.

    Both description and type hints above assume an array input for simplicity but this function also accepts Container instances in place of any arguments.

    Examples:

    With class:`ivy.Array`
    ----------------------

    >>> x= ivy.array([3 ,2 ,3 ,8 ,6])
    >>> y=ivy.argpartition(x,kth=0)
    >>> print(y)

    ivy.array([1.,2.,0.,4.,5])


    With class:`ivy.Container`
    --------------------------

    >>> x = ivy.Container(a= np.array([[30,-10],[10,-20]]), b=np.array([[25,-15],[15,-25]]))
    >>> y=x.argpartition(kth=[(-1,None)])
    >>>print(y)

    {
    "b":array([[0,1],[0 ,-2]])
    "c":array([[0.-8],[-7.-9]])}
    }

    """
    return ivy.current_backend(a).argpartition(
        a,
        kth,
        axis=axis,
        kind='introspect',
        order=None
        )



