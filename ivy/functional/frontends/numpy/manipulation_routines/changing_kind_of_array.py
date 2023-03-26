# local
import ivy
from typing import Optional, Union


def asanyarray(
    a: Union[ivy.Array],
    dtype: Optional[ivy.dtype],
    order: Optional = None,
    *,
    keepdims: bool = False,
    like=None
) -> ivy.Array:
    """
    Convert the input to an ndarray, but pass ndarray subclasses through.
    Parameters
    ----------
    a
        array_like
        Input data, in any form that can be converted to an array.
    dtype
        data-type, optional
        By default, the data-type is inferred from the input data.
    order
        {‘C’, ‘F’, ‘A’, ‘K’}, optional
        Memory layout. Defaults to ‘C’.
    like
        array_like, optional
        Reference object to allow the creation of arrays which are not NumPy arrays.
        If an array-like passed in as like supports the __array_function__ protocol,
    Returns
    -------
        out -- ndarray or an ndarray subclass
    """
    return ivy.any(a, axis=order, keepdims=keepdims, out=like)
