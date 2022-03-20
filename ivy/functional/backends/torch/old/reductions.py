"""
Collection of PyTorch reduction functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch
from typing import Optional, List


def einsum(equation, *operands):
    return _torch.einsum(equation, *operands)