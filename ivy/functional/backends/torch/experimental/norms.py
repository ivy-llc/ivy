import torch
from typing import Optional

from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def l2_normalize(
    x: torch.Tensor, /, *, axis: int = None, out: torch.Tensor = None
) -> torch.Tensor:

    return torch.nn.functional.normalize(x, p=2, dim=axis, out=out)


l2_normalize.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
def instance_norm(
    x: torch.Tensor,
    /,
    *,
    scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[torch.Tensor] = None,
    running_stddev: Optional[torch.Tensor] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
):
    if scale is not None:
        scale = torch.reshape(scale, shape=(1, -1, 1, 1))
    if bias is not None:
        bias = torch.reshape(bias, shape=(1, -1, 1, 1))
    if running_mean is not None:
        running_mean = torch.reshape(running_mean, shape=(1, -1, 1, 1))
    if running_stddev is not None:
        running_stddev = torch.reshape(running_stddev, shape=(1, -1, 1, 1))
    if data_format == "NHWC":
        x = torch.permute(x, (0, 3, 1, 2))
    elif data_format != "NCHW":
        raise NotImplementedError
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    if scale is None:
        scale = torch.ones_like(var)
    if bias is None:
        bias = torch.zeros_like(mean)
    if affine:
        if scale is None:
            scale = torch.ones_like(var)
        if bias is None:
            bias = torch.zeros_like(mean)
        normalized = scale * normalized + bias
    if track_running_stats:
        if running_mean is None:
            running_mean = torch.ones_like(mean)
        if running_stddev is None:
            running_stddev = torch.ones_like(var)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_stddev = momentum * running_stddev + (1 - momentum) * torch.sqrt(var)
        if data_format == "NHWC":
            normalized = torch.permute(normalized, (0, 2, 3, 1))
            running_mean = torch.permute(running_mean, (0, 2, 3, 1))
            running_stddev = torch.permute(running_stddev, (0, 2, 3, 1))
        return normalized, running_mean, running_stddev
    if data_format == "NHWC":
        normalized = torch.permute(normalized, (0, 2, 3, 1))
    return normalized


instance_norm.support_native_out = False
