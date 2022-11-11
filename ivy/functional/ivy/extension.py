"""FFT API Extension"""

# global
from typing import Optional, Tuple, Union, List, Callable

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
)
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
def fft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    /,
    *,
    norm: str="backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the one dimensional discrete Fourier transform given input at least 1-D input x.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*, where d_in indicates the dimension that needs FFT.
    dim
        The dimension along which to take the one dimensional FFT.
    norm
        Optional argument, "backward", "ortho" or "forward". Defaults to be "backward".
        "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$.
        "forward" indicates normalization by $\frac{1}{n}$.
    n
        Optional argument indicating the sequence length, if given, the input would be
        padded with zero or truncated to length n before performing FFT.\
        Should be a integer greater than 1.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the FFT operation.

    Examples
    --------

    >>> x = ivy.TODO

    >>> ivy.fft(x,1,32)

    """
    return current_backend(x).fft(
        x,dim,norm,n,out=out
    )
