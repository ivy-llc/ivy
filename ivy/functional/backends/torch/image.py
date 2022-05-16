"""Collection of PyTorch image functions, wrapped to fit Ivy syntax and signature."""

# global
from ctypes import Union
import math
import torch
from operator import mul as _mul
from functools import reduce as _reduce
from typing import List, Tuple, Optional

# local
from ivy.functional.backends import torch as _ivy


def stack_images(
    images: List[torch.Tensor], desired_aspect_ratio: List[int] = (1, 1)
) -> torch.Tensor:
    num_images = len(images)
    if num_images == 0:
        raise Exception("At least 1 image must be provided")
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
        images_to_concat = images[i * stack_height_int : (i + 1) * stack_height_int]
        images_to_concat += [torch.zeros_like(images[0])] * (
            stack_height_int - len(images_to_concat)
        )
        image_rows.append(torch.cat(images_to_concat, num_batch_dims))
    return torch.cat(image_rows, num_batch_dims + 1)


def linear_resample(x, num_samples: int, axis: int = -1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    num_vals = x_shape[axis]
    axis = axis % num_x_dims
    if axis != num_x_dims - 1:
        x_pre_shape = x_shape[0:axis] + x_shape[-1:] + x_shape[axis + 1 : -1]
        x = torch.swapaxes(x, axis, -1)
    else:
        x_pre_shape = x_shape[:-1]
    x = torch.reshape(x, ([-1, 1] + [num_vals]))
    ret = torch.nn.functional.interpolate(
        x, num_samples, mode="linear", align_corners=True
    )
    ret = torch.reshape(ret, x_pre_shape + [num_samples])
    if axis != num_x_dims - 1:
        return torch.transpose(ret, -1, axis)
    return ret


# noinspection PyUnresolvedReferences
def bilinear_resample(x, warp):
    batch_shape = x.shape[:-3]
    input_image_dims = x.shape[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    batch_shape_product = _reduce(_mul, batch_shape, 1)
    warp_flat = warp.view([batch_shape_product] + [-1, 1] + [2])
    warp_flat_x = 2 * warp_flat[..., 0:1] / (input_image_dims[1] - 1) - 1
    warp_flat_y = 2 * warp_flat[..., 1:2] / (input_image_dims[0] - 1) - 1
    warp_flat_scaled = torch.cat((warp_flat_x, warp_flat_y), -1)
    mat_flat = x.view([batch_shape_product] + input_image_dims + [-1])
    mat_flat_transposed = mat_flat.permute((0, 3, 1, 2))
    interpolated_flat_transposed = torch.nn.functional.grid_sample(
        mat_flat_transposed, warp_flat_scaled, align_corners=True
    )
    interpolated_flat = interpolated_flat_transposed.permute((0, 2, 3, 1))
    return interpolated_flat.view(batch_shape + [-1, num_feats])


def gradient_image(
    x, batch_shape: Optional[List[int]] = None, image_dims: Optional[List[int]] = None
):
    x_shape = x.shape
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    device = x.device
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
    dy = _ivy.concat(
        (dy, torch.zeros(batch_shape + [1, image_dims[1], num_dims], device=device)), -3
    )
    # noinspection PyTypeChecker
    dx = _ivy.concat(
        (dx, torch.zeros(batch_shape + [image_dims[0], 1, num_dims], device=device)), -2
    )
    # BS x H x W x D,    BS x H x W x D
    return dy, dx


def random_crop(
    x: torch.tensor, 
    crop_size: Tuple[int, int], 
    batch_shape: Optional[List[int]] = None, 
    image_dims: Optional[List[int]] = None
) -> torch.tensor:
    x_shape = x.shape
    if batch_shape is None:
        batch_shape = x_shape[:-3]
    if image_dims is None:
        image_dims = x_shape[-3:-1]
    num_channels = x_shape[-1]
    flat_batch_size = _reduce(_mul, batch_shape, 1)

    # shapes as list
    crop_size = list(crop_size)
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    margins = [img_dim - cs for img_dim, cs in zip(image_dims, crop_size)]

    # FBS x H x W x F
    x_flat = _ivy.reshape(x, [flat_batch_size] + image_dims + [num_channels])

    # FBS x 1
    x_offsets = torch.randint(0, margins[0] + 1, [flat_batch_size]).tolist()
    y_offsets = torch.randint(0, margins[1] + 1, [flat_batch_size]).tolist()

    # list of 1 x NH x NW x F
    cropped_list = [
        img[..., xo : xo + crop_size[0], yo : yo + crop_size[1], :]
        for img, xo, yo in zip(_ivy.unstack(x_flat, 0, True), x_offsets, y_offsets)
    ]

    # FBS x NH x NW x F
    flat_cropped = _ivy.concat(cropped_list, 0)

    # BS x NH x NW x F
    return _ivy.reshape(flat_cropped, batch_shape + crop_size + [num_channels])
