# global
from typing import Optional
import paddle
import paddle.nn.functional as F

# local
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    reduction: Optional[str] = "mean",
) -> paddle.Tensor:
    return F.l1_loss(input, target, reduction=reduction)


@with_supported_dtypes({"2.5.1 and below": ("float", "uint16")}, backend_version)
def smooth_l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
) -> paddle.Tensor:
    return paddle.nn.functional.smooth_l1_loss(
        input, target, reduction=reduction, beta=beta
    )
