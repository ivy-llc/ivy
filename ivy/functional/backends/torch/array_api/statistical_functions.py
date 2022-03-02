import torch

from typing import Union, Tuple, Optional, List


def var(x: torch.Tensor,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False) -> torch.Tensor:

    return torch.var(x, dim=axis, keepdim=keepdims)
