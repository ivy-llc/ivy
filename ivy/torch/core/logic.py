"""
Collection of PyTorch logic functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch

logical_and = lambda x1, x2: x1.type(_torch.bool) & x2.type(_torch.bool)
logical_or = lambda x1, x2: x1.type(_torch.bool) | x2.type(_torch.bool)
logical_not = lambda x: _torch.logical_not(x.type(_torch.bool))
