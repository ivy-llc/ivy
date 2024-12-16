import torch
import math
import itertools

from kornia.utils.one_hot import one_hot


def kornia_one_hot():
    labels = torch.tensor([[[0, 1], [2, 0]]])
    return one_hot(labels, num_classes=3, device=torch.device("cpu"), dtype=torch.int64)


def already_canonicalized_operations(data):
    # Already canonicalized operations
    sin_data = math.sin(data)
    sqrt_data = math.sqrt(data)
    chained_data = itertools.chain(data, sin_data, sqrt_data)
    return chained_data
