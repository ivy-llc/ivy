import torch

from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def l2_normalize(
    x: torch.Tensor, /, *, axis: int = None, out: torch.Tensor = None
) -> torch.Tensor:

    return torch.nn.functional.normalize(x, p=2, dim=axis, out=out)


l2_normalize.support_native_out = True
