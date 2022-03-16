# global
import torch
from torch import Tensor
from typing import Union, Tuple, Optional, Dict

# local
from ivy import dtype_from_str, default_dtype, dev_from_str, default_device
from ivy.functional.backends.torch.device import _callable_dev


def zeros(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.zeros(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))


def ones(shape: Union[int, Tuple[int]],
         dtype: Optional[torch.dtype] = None,
         device: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    dtype_val: torch.dtype = dtype_from_str(dtype)
    dev = default_device(device)
    return torch.ones(shape, dtype=dtype_val, device=dev_from_str(dev))


def ones_like(x : torch.Tensor,
              dtype: Optional[Union[torch.dtype, str]] = None,
              dev: Optional[Union[torch.device, str]] = None) \
        -> torch.Tensor:
    if dev is None:
        dev = _callable_dev(x)
    if dtype is not None and dtype is str:
        type_dict: Dict[str, torch.dtype] = {'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'bool': torch.bool}
        return torch.ones_like(x, dtype=type_dict[dtype], device=dev_from_str(dev))
    else:
        return torch.ones_like(x, dtype= dtype, device=dev_from_str(dev))

    return torch.ones_like(x, device=dev_from_str(dev))

  
def tril(x: torch.Tensor,
         k: int = 0) \
         -> torch.Tensor:
    return torch.tril(x, diagonal=k)


def triu(x: torch.Tensor,
         k: int = 0) \
         -> torch.Tensor:
    return torch.triu(x, diagonal=k)
    

def empty(shape: Union[int, Tuple[int]],
          dtype: Optional[torch.dtype] = None,
          device: Optional[torch.device] = None) \
        -> Tensor:
    return torch.empty(shape, dtype=dtype_from_str(default_dtype(dtype)), device=dev_from_str(default_device(device)))
