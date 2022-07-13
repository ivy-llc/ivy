# global
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)

import abc

# ToDo: implement all methods here as public instance methods
import ivy.array


class ArrayWithSearching(abc.ABC):
    pass

def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return abc.argmax(x, axis=axis, out=out, keepdims=keepdims)

    """Returns the indices of the maximum values along a specified axis. When the
    maximum value occurs multiple times, only the indices corresponding to the first
    occurrence are returned.

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis along which to search. If None, the function must return the index of the
        maximum value of the flattened array. Default  None.
    keepdims
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the array.
    out
        If provided, the result will be inserted into this array. It should be of the
        appropriate shape and dtype.

    Returns
    -------
    ret
        if axis is None, a zero-dimensional array containing the index of the first
        occurrence of the maximum value; otherwise, a non-zero-dimensional array
        containing the indices of the maximum values. The returned array must have be
        the default array index data type.
    
    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_.
    
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.



    Functional Examples
        --------

        With :code:`ivy.Array` input:

        >>> x = ivy.array([-0., 1., -1.])
        >>> y = ivy.argmax(x)
        >>> print(y)
        ivy.array([1])

        >>> x = ivy.array([-0., 1., -1.])
        >>> ivy.argmax(x,out=x)
        >>> print(x)
        ivy.array([1])

        >>> x=ivy.array([[1., -0., -1.], \
                         [-2., 3., 2.]])
        >>> y = ivy.argmax(x, axis= 1)
        >>> print(y)
        ivy.array([0, 1])
        
        >>> x = ivy.array([-0., 1., -1.])
        >>> y = ivy.argmax(x, axis)
        >>> print(y)
        ivy.array([1])
        
        >>> x = ivy.array([-0., 1., -1.])
        >>> y = ivy.argmax(x, axis, keepdims)
        >>> print(y)
        ivy.array([1])
        
        >>> x = ivy.array([-0., 1., -1.])
        >>> y = ivy.argmax(x, axis, keepdims= False)
        >>> print(y)
        ivy.array([1])

        >>> x=ivy.array([[4., 0., -1.], \
                         [2., -3., 6]])
        >>> y = ivy.argmax(x, axis= 1, keepdims= True)
        >>> print(y)
        ivy.array([[0], \
                  [2]])

        >>> x=ivy.array([[4., 0., -1.], \
                         [2., -3., 6], \
                         [2., -3., 6]])
        >>> z= ivy.zeros((1,3), dtype=ivy.int64)
        >>> y = ivy.argmax(x, axis= 1, keepdims= True, out= z)
        >>> print(z)
        ivy.array([[0], \
                   [2], \
                   [2]])

        With :code:`ivy.NativeArray` input:

        >>> x = ivy.native_array([-0., 1., -1.])
        >>> y = ivy.argmax(x)
        >>> print(y)
        ivy.array([1])

        Instance Method Examples
        ------------------------

        Using :code:`ivy.Array` instance method:

        >>> x = ivy.array([0., 1., 2.])
        >>> y = x.argmax()
        >>> print(y)
        ivy.array(2)

        """
