# global
from typing import Optional, Union, Tuple
import torch

# local
import ivy


def fft(
    x: torch.Tensor,
    dim: int,
    /,
    *,
    norm: str="backward",
    n: Union[int, Tuple[int]] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
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
    return torch.fft.fft(x,n,dim,norm,out=out)
