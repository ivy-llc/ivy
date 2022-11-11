# global

from typing import Optional, Union, Tuple

# local
import ivy
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def fft(
    x: JaxArray,
    dim: int,
    /,
    *,
    norm: str="backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if not isinstance(dim,int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(dim)}")
    if n < -len(x.shape) :
        raise ivy.exceptions.IvyError(f"Invalid dim {dim}, expecting ranging from {-len(x.shape)} to {len(x.shape)-1}  ")
    if n is None:
        n = x.shape[dim]
    if not isinstance(n,int):
        raise ivy.exceptions.IvyError(f"Expecting <class 'int'> instead of {type(n)}")
    if n <= 1 :
        raise ivy.exceptions.IvyError(f"Invalid data points {n}, expecting more than 1")
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.exceptions.IvyError(f"Unrecognized normalization mode {norm}")
    return jnp.fft(x,n,dim,norm)
