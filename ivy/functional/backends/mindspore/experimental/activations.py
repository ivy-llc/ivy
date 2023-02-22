from typing import Optional, Union

# global
import mindspore as ms
import mindspore.numpy as np

# local


def logit(x: ms.Tensor, /, *, eps: Optional[float] = None, out=None):
    return ms.logit(x, eps=eps)


def thresholded_relu(
    x: ms.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return np.where(x > threshold, x, 0)
