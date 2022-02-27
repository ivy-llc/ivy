# global
import torch
from torch import Tensor
from typing import Union, Tuple, Optional

# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device


def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.empty(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))
