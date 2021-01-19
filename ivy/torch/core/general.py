"""
Collection of PyTorch general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    dtype = _torch.__dict__[dtype_str] if dtype_str else dtype_str
    return _torch.tensor(object_in, dtype=dtype).to(dev)
