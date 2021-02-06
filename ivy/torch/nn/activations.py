"""
Collection of PyTorch activation functions, wrapped to fit Ivy syntax and signature.
"""

# global
import torch as _torch

relu = _torch.nn.functional.relu
leaky_relu = _torch.nn.functional.leaky_relu
tanh = _torch.nn.functional.tanh
sigmoid = _torch.nn.functional.sigmoid
softmax = _torch.nn.functional.softmax
softplus = _torch.nn.functional.softplus
