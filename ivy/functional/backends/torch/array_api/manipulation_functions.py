# global
from typing import Union, Tuple
import torch


def roll(x: torch.Tensor, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None)\
      -> torch.Tensor:
   return torch.roll(x, shift, axis) 
