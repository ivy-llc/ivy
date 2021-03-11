"""
Collection of PyTorch image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
import torch
from operator import mul
from functools import reduce
from typing import List, Optional

# local
from ivy import torch as _ivy


def stack_images(images: List[torch.Tensor], desired_aspect_ratio:List[int]=(1, 1),
                 batch_shape:Optional[List[int]]=None, image_dims:Optional[List[int]]=None):
    num_images = len(images)
    if num_images == 0:
        raise Exception('At least 1 image must be provided')
    if batch_shape is None:
        batch_shape = _ivy.shape(images[0])[:-3]
    if image_dims is None:
        image_dims = _ivy.shape(images[0])[-3:-1]
    num_batch_dims = len(batch_shape)
    if num_images == 1:
        return images[0]
    img_ratio = image_dims[0]/image_dims[1]
    desired_img_ratio = desired_aspect_ratio[0]/desired_aspect_ratio[1]
    stack_ratio = img_ratio*desired_img_ratio
    stack_height = (num_images/stack_ratio)**0.5
    stack_height_int = math.ceil(stack_height)
    stack_width_int:int = math.ceil(num_images/stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i*stack_height_int:(i+1)*stack_height_int]
        images_to_concat += [_ivy.zeros_like(images[0])] * (stack_height_int - len(images_to_concat))
        image_rows.append(_ivy.concatenate(images_to_concat, num_batch_dims))
    return _ivy.concatenate(image_rows, num_batch_dims + 1)


# noinspection PyUnresolvedReferences
def bilinear_resample(x, warp, batch_shape:Optional[List[int]]=None, input_image_dims:Optional[List[int]]=None,
                      output_image_dims:Optional[List[int]]=None):
    if batch_shape is None:
        batch_shape = _ivy.shape(x)[:-3]
    if input_image_dims is None:
        input_image_dims = _ivy.shape(x)[-3:-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    batch_shape_product = reduce(mul, batch_shape, 1)
    try:
        warp_flat = warp.view([batch_shape_product] + input_image_dims + [2])
    except:
        d = 0
    warp_flat_x = 2 * warp_flat[:, :, :, 0:1] / (input_image_dims[1] - 1) - 1
    warp_flat_y = 2 * warp_flat[:, :, :, 1:2] / (input_image_dims[0] - 1) - 1
    warp_flat_scaled = torch.cat((warp_flat_x, warp_flat_y), -1)
    mat_flat = x.view([batch_shape_product] + input_image_dims + [-1])
    mat_flat_transposed = mat_flat.permute((0, 3, 1, 2))
    interpolated_flat_transposed = torch.nn.functional.grid_sample(mat_flat_transposed, warp_flat_scaled, align_corners=True)
    interpolated_flat = interpolated_flat_transposed.permute((0, 2, 3, 1))
    return interpolated_flat.view(batch_shape + input_image_dims + [-1])


def gradient_image(x, batch_shape:Optional[List[int]]=None, image_dims:Optional[List[int]]=None,
                   dev:Optional[torch.device]=None):
    x_shape = _ivy.shape(x)
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    if dev is None:
        dev = _ivy.dev_str(x)
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    dy = _ivy.concatenate((dy, _ivy.zeros(batch_shape + [1, image_dims[1], num_dims], dev=dev)), -3)
    dx = _ivy.concatenate((dx, _ivy.zeros(batch_shape + [image_dims[0], 1, num_dims], dev=dev)), -2)
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
