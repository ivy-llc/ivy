"""
Collection of PyTorch logic functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


def logical_and(x1, x2):
    return x1.type(_torch.bool) & x2.type(_torch.bool)


def logical_or(x1, x2):
    return x1.type(_torch.bool) | x2.type(_torch.bool)


def logical_not(x):
    return _torch.logical_not(x.type(_torch.bool))
