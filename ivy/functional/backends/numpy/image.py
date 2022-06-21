"""Collection of Numpy image functions, wrapped to fit Ivy syntax and signature."""

# global
import math
import numpy as np
from operator import mul
from functools import reduce
from typing import List, Tuple

# local
from ivy.functional.backends import numpy as ivy


def stack_images(
    images: List[np.ndarray], desired_aspect_ratio: Tuple[int, int] = (1, 1)
) -> np.ndarray:
    num_images = len(images)
    if num_images == 0:
        raise Exception("At least 1 image must be provided")
    batch_shape = ivy.shape(images[0])[:-3]
    image_dims = ivy.shape(images[0])[-3:-1]
    num_batch_dims = len(batch_shape)
    if num_images == 1:
        return images[0]
    img_ratio = image_dims[0] / image_dims[1]
    desired_img_ratio = desired_aspect_ratio[0] / desired_aspect_ratio[1]
    stack_ratio = img_ratio * desired_img_ratio
    stack_height = (num_images / stack_ratio) ** 0.5
    stack_height_int = math.ceil(stack_height)
    stack_width_int = math.ceil(num_images / stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i * stack_height_int : (i + 1) * stack_height_int]
        images_to_concat += [ivy.zeros_like(images[0])] * (
            stack_height_int - len(images_to_concat)
        )
        image_rows.append(ivy.concat(images_to_concat, num_batch_dims))
    return ivy.concat(image_rows, num_batch_dims + 1)


def linear_resample(x, num_samples, axis=-1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    x_pre_shape = x_shape[0:axis]
    x_pre_size = reduce(mul, x_pre_shape) if x_pre_shape else 1
    num_pre_dims = len(x_pre_shape)
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis + 1 :]
    x_post_size = reduce(mul, x_post_shape) if x_post_shape else 1
    num_post_dims = len(x_post_shape)
    xp = np.reshape(np.arange(num_vals * x_pre_size * x_post_size), x_shape)
    x_coords = (
        np.arange(num_samples) * ((num_vals - 1) / (num_samples - 1)) * x_post_size
    )
    x_coords = np.reshape(
        x_coords, [1] * num_pre_dims + [num_samples] + [1] * num_post_dims
    )
    x_coords = np.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = np.reshape(x, (-1,))
    xp = np.reshape(xp, (-1,))
    x_coords = np.reshape(x_coords, (-1,))
    ret = np.interp(x_coords, xp, x)
    return np.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


# noinspection PyPep8Naming
def bilinear_resample(x, warp):
    batch_shape = x.shape[:-3]
    input_image_dims = x.shape[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    # image statistics
    height, width = input_image_dims
    max_x = width - 1
    max_y = height - 1
    idx_size = warp.shape[-2]
    batch_shape_flat = int(np.prod(np.asarray(batch_shape)))
    # B
    batch_offsets = np.arange(batch_shape_flat) * height * width
    # B x (HxW)
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, idx_size])
    # (BxHxW)
    base = np.reshape(base_grid, [-1])
    # (BxHxW) x D
    data_flat = np.reshape(x, [batch_shape_flat * height * width, -1])
    # (BxHxW) x 2
    warp_flat = np.reshape(warp, [-1, 2])
    warp_floored = (np.floor(warp_flat)).astype(np.int32)
    bilinear_weights = warp_flat - np.floor(warp_flat)
    # (BxHxW)
    x0 = warp_floored[:, 0]
    x1 = x0 + 1
    y0 = warp_floored[:, 1]
    y1 = y0 + 1
    x0 = np.clip(x0, 0, max_x)
    x1 = np.clip(x1, 0, max_x)
    y0 = np.clip(y0, 0, max_y)
    y1 = np.clip(y1, 0, max_y)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    # (BxHxW) x D
    Ia = np.take(data_flat, idx_a, axis=0)
    Ib = np.take(data_flat, idx_b, axis=0)
    Ic = np.take(data_flat, idx_c, axis=0)
    Id = np.take(data_flat, idx_d, axis=0)
    # (BxHxW)
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]
    # (BxHxW) x 1
    wa = np.expand_dims((1 - xw) * (1 - yw), 1)
    wb = np.expand_dims((1 - xw) * yw, 1)
    wc = np.expand_dims(xw * (1 - yw), 1)
    wd = np.expand_dims(xw * yw, 1)
    # (BxNP) x D
    resampled_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # B x NP x D
    return np.reshape(resampled_flat, batch_shape + [-1, num_feats])


def gradient_image(x):
    x_shape = ivy.shape(x)
    batch_shape = x_shape[:-3]
    image_dims = x_shape[-3:-1]
    device = ivy.dev(x)
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    dy = ivy.concat(
        (dy, ivy.zeros(batch_shape + [1, image_dims[1], num_dims], device=device)), -3
    )
    dx = ivy.concat(
        (dx, ivy.zeros(batch_shape + [image_dims[0], 1, num_dims], device=device)), -2
    )
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
