import torch
from typing import Optional, Tuple

from ivy.func_wrapper import with_unsupported_dtypes, handle_mixed_function
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


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, backend_version)
@handle_mixed_function(
    lambda x, mean, variance, scale, offset, **kwargs: (
        x.ndim > 1
        and mean.ndim == 1
        and variance.ndim == 1
        and (scale is None or scale.ndim == 1)
        and (offset is None or offset.ndim == 1)
    )
)
def batch_norm(
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    /,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    training: bool = False,
    eps: float = 1e-5,
    momentum: float = 1e-1,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # reshape to N,C,H,W
    xdims = x.ndim
    x = torch.permute(x, dims=(0, xdims - 1, *range(1, xdims - 1)))
    mean.requires_grad = False
    variance.requires_grad = False
    scale.requires_grad = False
    offset.requires_grad = False
    runningmean = mean.clone()
    runningvariance = variance.clone()
    xnormalized = torch.nn.functional.batch_norm(
        x,
        runningmean,
        runningvariance,
        weight=scale,
        bias=offset,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    xnormalized = torch.permute(xnormalized, dims=(0, *range(2, xdims), 1))
    return xnormalized, runningmean, runningvariance


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
@handle_mixed_function(
    lambda x, mean, variance, scale, offset, **kwargs: (
        x.ndim > 1
        and mean.ndim == 1
        and variance.ndim == 1
        and (scale is None or scale.ndim == 1)
        and (offset is None or offset.ndim == 1)
    )
)
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
    momentum: float = 1e-1,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean.requires_grad = False
    variance.requires_grad = False
    scale.requires_grad = False
    offset.requires_grad = False
    runningmean = mean.clone()
    runningvariance = variance.clone()
    # reshape  from  N, *S, C to N, C, *S
    xdims = x.ndim
    x = torch.permute(x, dims=(0, xdims - 1, *range(1, xdims - 1)))

    xnormalized = torch.nn.functional.instance_norm(
        x,
        runningmean,
        runningvariance,
        weight=scale,
        bias=offset,
        use_input_stats=training,
        eps=eps,
        momentum=momentum,
    )
    xnormalized = torch.permute(xnormalized, dims=(0, *range(2, xdims), 1))
    return xnormalized, runningmean, runningvariance


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
