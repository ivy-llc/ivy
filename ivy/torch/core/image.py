"""
Collection of PyTorch image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
import torch as _torch
from operator import mul
from functools import reduce
from typing import List, Optional

# local
from ivy import torch as _ivy


def stack_images(images: List[_torch.Tensor], desired_aspect_ratio: List[int] = (1, 1)):
    num_images = len(images)
    if num_images == 0:
        raise Exception('At least 1 image must be provided')
    batch_shape = images[0].shape[:-3]
    image_dims = images[0].shape[-3:-1]
    num_batch_dims = len(batch_shape)
    if num_images == 1:
        return images[0]
    img_ratio = image_dims[0] / image_dims[1]
    desired_img_ratio = desired_aspect_ratio[0] / desired_aspect_ratio[1]
    stack_ratio = img_ratio * desired_img_ratio
    stack_height = (num_images / stack_ratio) ** 0.5
    stack_height_int = math.ceil(stack_height)
    stack_width_int: int = math.ceil(num_images / stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i * stack_height_int:(i + 1) * stack_height_int]
        images_to_concat += [_torch.zeros_like(images[0])] * (stack_height_int - len(images_to_concat))
        image_rows.append(_torch.cat(images_to_concat, num_batch_dims))
    return _torch.cat(image_rows, num_batch_dims + 1)


# noinspection PyUnresolvedReferences
def bilinear_resample(x, warp):
    batch_shape = x.shape[:-3]
    input_image_dims = x.shape[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    batch_shape_product = reduce(mul, batch_shape, 1)
    warp_flat = warp.view([batch_shape_product] + [-1, 1] + [2])
    warp_flat_x = 2 * warp_flat[..., 0:1] / (input_image_dims[1] - 1) - 1
    warp_flat_y = 2 * warp_flat[..., 1:2] / (input_image_dims[0] - 1) - 1
    warp_flat_scaled = _torch.cat((warp_flat_x, warp_flat_y), -1)
    mat_flat = x.view([batch_shape_product] + input_image_dims + [-1])
    mat_flat_transposed = mat_flat.permute((0, 3, 1, 2))
    interpolated_flat_transposed = _torch.nn.functional.grid_sample(mat_flat_transposed, warp_flat_scaled,
                                                                    align_corners=True)
    interpolated_flat = interpolated_flat_transposed.permute((0, 2, 3, 1))
    return interpolated_flat.view(batch_shape + [-1, num_feats])


def gradient_image(x, batch_shape: Optional[List[int]] = None, image_dims: Optional[List[int]] = None):
    x_shape = x.shape
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    dev = x.device
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    # noinspection PyTypeChecker
    dy = _ivy.concatenate((dy, _torch.zeros(batch_shape + [1, image_dims[1], num_dims], device=dev)), -3)
    # noinspection PyTypeChecker
    dx = _ivy.concatenate((dx, _torch.zeros(batch_shape + [image_dims[0], 1, num_dims], device=dev)), -2)
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
