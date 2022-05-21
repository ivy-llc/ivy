"""Collection of Jax image functions, wrapped to fit Ivy syntax and signature."""

# global
import math
import jax
import jax.numpy as jnp
from operator import mul as _mul
from functools import reduce as _reduce

# local
from ivy.functional.backends import jax as _ivy


def stack_images(images, desired_aspect_ratio=(1, 1)):
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
    xp = jnp.reshape(jnp.arange(num_vals * x_pre_size * x_post_size), x_shape)
    x_coords = (
        jnp.arange(num_samples) * ((num_vals - 1) / (num_samples - 1)) * x_post_size
    )
    x_coords = jnp.reshape(
        x_coords, [1] * num_pre_dims + [num_samples] + [1] * num_post_dims
    )
    x_coords = jnp.broadcast_to(x_coords, x_pre_shape + [num_samples] + x_post_shape)
    slc = [slice(None)] * num_x_dims
    slc[axis] = slice(0, 1, 1)
    x_coords = x_coords + xp[tuple(slc)]
    x = jnp.reshape(x, (-1,))
    xp = jnp.reshape(xp, (-1,))
    x_coords = jnp.reshape(x_coords, (-1,))
    ret = jnp.interp(x_coords, xp, x)
    return jnp.reshape(ret, x_pre_shape + [num_samples] + x_post_shape)


# noinspection PyPep8Naming
def bilinear_resample(x, warp):
    batch_shape = x.shape[:-3]
    input_image_dims = x.shape[-3:-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    num_feats = x.shape[-1]
    # image statistics
    height, width = input_image_dims
    max_x = width - 1
    max_y = height - 1
    idx_size = _reduce(_mul, warp.shape[-3:-1], 1)
    batch_shape_flat = _reduce(_mul, batch_shape, 1)
    # B
    batch_offsets = jnp.arange(batch_shape_flat) * idx_size
    # B x (HxW)
    base_grid = jnp.tile(jnp.expand_dims(batch_offsets, 1), [1, idx_size])
    # (BxHxW)
    base = jnp.reshape(base_grid, [-1])
    # (BxHxW) x D
    data_flat = jnp.reshape(x, [batch_shape_flat * height * width, -1])
    # (BxHxW) x 2
    warp_flat = jnp.reshape(warp, [-1, 2])
    warp_floored = (jnp.floor(warp_flat)).astype(jnp.int32)
    bilinear_weights = warp_flat - jnp.floor(warp_flat)
    # (BxHxW)
    x0 = warp_floored[:, 0]
    x1 = x0 + 1
    y0 = warp_floored[:, 1]
    y1 = y0 + 1
    x0 = jnp.clip(x0, 0, max_x)
    x1 = jnp.clip(x1, 0, max_x)
    y0 = jnp.clip(y0, 0, max_y)
    y1 = jnp.clip(y1, 0, max_y)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    # (BxHxW) x D
    Ia = jnp.take(data_flat, idx_a, axis=0)
    Ib = jnp.take(data_flat, idx_b, axis=0)
    Ic = jnp.take(data_flat, idx_c, axis=0)
    Id = jnp.take(data_flat, idx_d, axis=0)
    # (BxHxW)
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]
    # (BxHxW) x 1
    wa = jnp.expand_dims((1 - xw) * (1 - yw), 1)
    wb = jnp.expand_dims((1 - xw) * yw, 1)
    wc = jnp.expand_dims(xw * (1 - yw), 1)
    wd = jnp.expand_dims(xw * yw, 1)
    # (BxHxW) x D
    resampled_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # B x H x W x D
    return jnp.reshape(resampled_flat, batch_shape + [-1, num_feats])


def gradient_image(x):
    x_shape = _ivy.shape(x)
    batch_shape = x_shape[:-3]
    image_dims = x_shape[-3:-1]
    device = x.device_buffer.device()
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    # jax.device_put(x, as_native_dev(dev))
    dy = _ivy.concat(
        (
            dy,
            jax.device_put(
                jnp.zeros(batch_shape + [1, image_dims[1], num_dims]), device
            ),
        ),
        -3,
    )
    dx = _ivy.concat(
        (
            dx,
            jax.device_put(
                jnp.zeros(batch_shape + [image_dims[0], 1, num_dims]), device
            ),
        ),
        -2,
    )
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
