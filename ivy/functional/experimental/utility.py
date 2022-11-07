# global
from typing import Union, Optional, Sequence

# local
import ivy
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
)
from ivy.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def is_storage(x: Any)  -> bool:
    """Tests whether the object x is a storage
    Parameters
    ----------
    x
        Any input object. It can be String, Array etc
    Returns
    -------
    ret
        Returns ``bool`` datatype with True or False value
        Functional Examples
    -------------------
    With :code:`ivy.Array` input:
    >>> x = ivy.array([2, 3, 4])
    >>> y = ivy.is_storage(x)
    >>> print(y)
    False
    
    >>> x = ivy.array([[0],[1]])
    >>> y = ivy.is_storage(x)
    >>> print(y)
    True

    >>> x = ivy.array([])
    >>> y = ivy.is_storage(x)
    >>> print(y)
    False
    """
    return ivy.current_backend(x).is_storage(x)