"""Collection of MXNet image functions, wrapped to fit Ivy syntax and signature."""

import math
from functools import reduce as _reduce
from operator import mul as _mul
import mxnet as mx
from typing import List, Tuple

# local
from ivy.functional.backends import mxnet as _ivy


def stack_images(
    images: List[mx.nd.NDArray], desired_aspect_ratio: Tuple[int, int] = (1, 1)
) -> mx.nd.NDArray:
    num_images = len(images)
    if num_images == 0:
        raise Exception("At least 1 image must be provided")
    batch_shape = _ivy.shape(images[0])[:-3]
    image_dims = _ivy.shape(images[0])[-3:-1]
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
        images_to_concat += [_ivy.zeros_like(images[0])] * (
            stack_height_int - len(images_to_concat)
        )
        image_rows.append(_ivy.concat(images_to_concat, num_batch_dims))
    return _ivy.concat(image_rows, num_batch_dims + 1)


def linear_resample(x, num_samples, axis=-1):
    x_shape = list(x.shape)
    num_x_dims = len(x_shape)
    axis = axis % num_x_dims
    x_pre_shape = x_shape[0:axis]
    x_pre_size = _reduce(_mul, x_pre_shape) if x_pre_shape else 1
    num_pre_dims = len(x_pre_shape)
    num_vals = x.shape[axis]
    x_post_shape = x_shape[axis + 1 :]
    x_post_size = _reduce(_mul, x_post_shape) if x_post_shape else 1
    num_post_dims = len(x_post_shape)
    xp = mx.nd.reshape(mx.nd.arange(num_vals * x_pre_size * x_post_size), x_shape)
    x_coords = (
        mx.nd.arange(num_samples) * ((num_vals - 1) / (num_samples - 1)) * x_post_size
    )
    x_coords = mx.nd.reshape(
        x_coords, [1] * num_pre_dims + [num_samples] + [1] * num_post_dims
    )
    x_coords = mx.nd.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = mx.nd.reshape(x, (-1,))
    xp = mx.nd.reshape(xp, (-1,))
    x_coords = mx.nd.reshape(x_coords, (-1,))
    ret = mx.nd.array(mx.np.interp(x_coords.asnumpy(), xp.asnumpy(), x.asnumpy()))
    return mx.nd.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


def bilinear_resample(x, warp):
    batch_shape = _ivy.shape(x)[:-3]
    input_image_dims = _ivy.shape(x)[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    batch_shape_product = _reduce(_mul, batch_shape, 1)
    warp_flat = mx.nd.reshape(warp, [batch_shape_product] + [-1, 1] + [2])
    warp_flat_x = 2 * warp_flat[..., 0:1] / (input_image_dims[1] - 1) - 1
    warp_flat_y = 2 * warp_flat[..., 1:2] / (input_image_dims[0] - 1) - 1
    warp_flat_scaled = mx.nd.concat(warp_flat_x, warp_flat_y, dim=-1)
    warp_flat_trans = mx.nd.transpose(warp_flat_scaled, (0, 3, 1, 2))
    mat_flat = mx.nd.reshape(x, [batch_shape_product] + input_image_dims + [-1])
    mat_flat_trans = mx.nd.transpose(mat_flat, (0, 3, 1, 2))
    interpolated_flat_transposed = mx.nd.BilinearSampler(
        mat_flat_trans, warp_flat_trans
    )
    interpolated_flat = mx.nd.transpose(interpolated_flat_transposed, (0, 2, 3, 1))
    return mx.nd.reshape(interpolated_flat, batch_shape + [-1, num_feats])


def gradient_image(x):
    x_shape = _ivy.shape(x)
    batch_shape = x_shape[:-3]
    image_dims = x_shape[-3:-1]
    ctx = x.context
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
        (dy, mx.nd.zeros(batch_shape + [1, image_dims[1], num_dims], ctx=ctx)), -3
    )
    # noinspection PyTypeChecker
    dx = _ivy.concat(
        (dx, mx.nd.zeros(batch_shape + [image_dims[0], 1, num_dims], ctx=ctx)), -2
    )
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
