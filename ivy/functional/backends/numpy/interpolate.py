import ivy
import ivy.numpy as np

def interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor must be defined")
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor can be defined")
    if size is not None:
        height, width = size
        height_scale = float(height) / float(x.shape[2])
        width_scale = float(width) / float(x.shape[3])
    else:
        height_scale, width_scale = scale_factor
    if mode == 'nearest':
        x = ivy.resize_nearest_neighbor(x, (x.shape[2] * height_scale, x.shape[3] * width_scale))
    elif mode == 'linear':
        x = ivy.resize_linear(x, (x.shape[2] * height_scale, x.shape[3] * width_scale))
    else:
        raise ValueError("unsupported interpolation mode: {}".format(mode))
    return x
