



# global
import math
import torch as _torch


def cos(x: _torch.Tensor) -> _torch.Tensor:
    if isinstance(x, float):
        return math.cos(x)
    return _torch.cos(x)
