import torch
from kornia.filters.sobel import spatial_gradient, SpatialGradient


def kornia_spatial_gradient():
    args = (
        torch.rand((1, 1, 4, 4)),
        torch.eye(3).unsqueeze(0),
    )
    return spatial_gradient(*args)


def kornia_spatial_gradient_cls():
    args = (
        torch.rand((1, 1, 4, 4)),
        torch.eye(3).unsqueeze(0),
    )
    return SpatialGradient()(*args)
