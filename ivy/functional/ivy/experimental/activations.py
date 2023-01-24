# global
from typing import Union, Optional

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.exceptions import handle_exceptions
from ivy.func_wrapper import (
    handle_nestable,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
)


@handle_out_argument
@handle_nestable
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def logit(
    x: Union[float, int, ivy.Array],
    /,
    *,
    eps: Optional[float] = None,
    out: Optional["ivy.Array"] = None,
) -> ivy.Array:
    """
    Computes the logit of x, i.e. logit(x) = log(x / (1 - x)).

    Parameters
    ----------
    x
        Input data.
    eps
        When eps is None the function outpus NaN where x < 0 or x > 1.
        and inf or -inf where x = 1 or x = 0, respectively.
        Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
    out
        Optional output array.

    Returns
    -------
    ret
        Array containing elementwise logits of x.

    Examples
    --------
    >>> x = ivy.array([1, 0, 0.9])
    >>> z = ivy.logit(x)
    >>> print(z)
    ivy.array([       inf,       -inf, 2.19722438])

    >>> x = ivy.array([1, 2, -0.9])
    >>> z = ivy.logit(x, eps=0.2)
    >>> print(z)
    ivy.array([ 1.38629448,  1.38629448, -1.38629436])

    """
    return current_backend(x).logit(x, eps=eps, out=out)
