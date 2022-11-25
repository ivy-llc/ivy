from typing import Optional
import torch
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
def isin(
    elements: torch.tensor,
    test_elements: torch.tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> torch.tensor:
    return torch.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True
