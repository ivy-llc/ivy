# global
from typing import Optional, Union, Sequence
import torch
import ivy

# local
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


# dirichlet
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def dirichlet(
    alpha: Union[torch.tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    out: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    size = size if size is not None else len(alpha)
    if seed is not None:
        torch.manual_seed(seed)
    return torch.tensor(
        torch.distributions.dirichlet.Dirichlet(alpha).rsample(sample_shape=size),
        dtype=dtype,
    )
