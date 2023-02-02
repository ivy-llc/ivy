import numpy as np
from ivy.func_wrapper import with_unsupported_dtypes
from typing import Optional
from .. import backend_version


@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, backend_version)
def l2_normalize(x: np.ndarray, /, *, axis: int = None, out=None) -> np.ndarray:
    denorm = np.linalg.norm(x, axis=axis, ord=2, keepdims=True)
    denorm = np.maximum(denorm, 1e-12)
    return x / denorm


def instance_norm(
    x: np.ndarray,
    /,
    *,
    scale: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[np.ndarray] = None,
    running_stddev: Optional[np.ndarray] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
):
    if scale is not None:
        scale = np.expand_dims(scale, axis=(0, 2, 3))
    if bias is not None:
        bias = np.expand_dims(bias, axis=(0, 2, 3))
    if running_mean is not None:
        running_mean = np.expand_dims(running_mean, axis=(0, 2, 3))
    if running_stddev is not None:
        running_stddev = np.expand_dims(running_stddev, axis=(0, 2, 3))
    if data_format == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))
    elif data_format != "NCHW":
        raise NotImplementedError
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    if affine:
        if scale is None:
            scale = np.ones_like(var)
        if bias is None:
            bias = np.zeros_like(mean)
        normalized = scale * normalized + bias
    if track_running_stats:
        if running_mean is None:
            running_mean = np.zeros_like(mean)
        if running_stddev is None:
            running_stddev = np.ones_like(var)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_stddev = momentum * running_stddev + (1 - momentum) * np.sqrt(var)
        if data_format == "NHWC":
            normalized = np.transpose(normalized, (0, 2, 3, 1))
            running_mean = np.transpose(running_mean, (0, 2, 3, 1))
            running_stddev = np.transpose(running_stddev, (0, 2, 3, 1))
        return normalized, running_mean, running_stddev
    if data_format == "NHWC":
        normalized = np.transpose(normalized, (0, 2, 3, 1))
    return normalized
