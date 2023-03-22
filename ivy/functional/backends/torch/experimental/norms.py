import torch
from typing import Optional

from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def l2_normalize(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    return torch.nn.functional.normalize(x, p=2, dim=axis, out=out)


l2_normalize.support_native_out = True


@with_unsupported_dtypes({"0.11.0 and below": ("bfloat16", "float16")}, backend_version)
def batch_norm(
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    /,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    training: bool = False,
    eps: float = 0e-5,
    momentum: float = 1e-1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mean.requires_grad = False
    variance.requires_grad = False
    scale.requires_grad = False
    offset.requires_grad = False
    runningmean = mean.clone()
    runningvariance = variance.clone()
    result = torch.nn.functional.batch_norm(
        x,
        runningmean,
        runningvariance,
        weight=scale,
        bias=offset,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    return result, runningmean, runningvariance


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
def instance_norm(
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    /,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    training: bool = False,
    eps: float = 0e-5,
) -> torch.Tensor:
    return torch.nn.functional.instance_norm(
        x,
        mean,
        variance,
        weight=scale,
        bias=offset,
        use_input_stats=not training,
        eps=eps,
    )


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def lp_normalize(
    x: torch.Tensor,
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    return torch.nn.functional.normalize(x, p=p, dim=axis, out=out)


lp_normalize.support_native_out = True
