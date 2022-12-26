# global
from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_nestable,
    to_native_arrays_and_back,
    handle_array_like,
    handle_out_argument,
)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional['ivy.Array'] = None,
) -> ivy.Array:
    """
    Logit function.
    Parameters
    ----------
    x
        Input data.
    eps
        A small positive number to avoid division by zero.
    out
        Optional output array.
    """
    return current_backend(x).logit(x, eps=eps, out=out)
