import paddle
from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional


def l2_normalize(
    x: paddle.Tensor, /, *, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def instance_norm(
    x: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor],
    bias: Optional[paddle.Tensor],
    eps: float = 1e-05,
    momentum: Optional[float] = 0.1,
    data_format: str = "NCHW",
    running_mean: Optional[paddle.Tensor] = None,
    running_stddev: Optional[paddle.Tensor] = None,
    affine: Optional[bool] = True,
    track_running_stats: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
):
    raise IvyNotImplementedException()


def lp_normalize(
    x: paddle.Tensor, /, *, p: float = 2, axis: int = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
