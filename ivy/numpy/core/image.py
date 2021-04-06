"""
Collection of Numpy image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
import numpy as _np

# local
from ivy import numpy as _ivy


def stack_images(images, desired_aspect_ratio=(1, 1)):
    num_images = len(images)
    if num_images == 0:
        raise Exception('At least 1 image must be provided')
    batch_shape = _ivy.shape(images[0])[:-3]
    image_dims = _ivy.shape(images[0])[-3:-1]
    num_batch_dims = len(batch_shape)
    if num_images == 1:
        return images[0]
    img_ratio = image_dims[0]/image_dims[1]
    desired_img_ratio = desired_aspect_ratio[0]/desired_aspect_ratio[1]
    stack_ratio = img_ratio*desired_img_ratio
    stack_height = (num_images/stack_ratio)**0.5
    stack_height_int = math.ceil(stack_height)
    stack_width_int = math.ceil(num_images/stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i*stack_height_int:(i+1)*stack_height_int]
        images_to_concat += [_ivy.zeros_like(images[0])] * (stack_height_int - len(images_to_concat))
        image_rows.append(_ivy.concatenate(images_to_concat, num_batch_dims))
    return _ivy.concatenate(image_rows, num_batch_dims + 1)


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
    idx_size = _np.prod(_np.asarray(_ivy.shape(warp)[-3:-1]))
    batch_shape_flat = int(_np.prod(_np.asarray(batch_shape)))
    # B
    batch_offsets = _np.arange(batch_shape_flat) * idx_size
    # B x (HxW)
    base_grid = _np.tile(_np.expand_dims(batch_offsets, 1), [1, idx_size])
    # (BxHxW)
    base = _np.reshape(base_grid, [-1])
    # (BxHxW) x D
    data_flat = _np.reshape(x, [batch_shape_flat * height * width, -1])
    # (BxHxW) x 2
    warp_flat = _np.reshape(warp, [-1, 2])
    warp_floored = (_np.floor(warp_flat)).astype(_np.int32)
    bilinear_weights = warp_flat - _np.floor(warp_flat)
    # (BxHxW)
    x0 = warp_floored[:, 0]
    x1 = x0 + 1
    y0 = warp_floored[:, 1]
    y1 = y0 + 1
    x0 = _np.clip(x0, 0, max_x)
    x1 = _np.clip(x1, 0, max_x)
    y0 = _np.clip(y0, 0, max_y)
    y1 = _np.clip(y1, 0, max_y)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1
    # (BxHxW) x D
    Ia = _np.take(data_flat, idx_a, axis=0)
    Ib = _np.take(data_flat, idx_b, axis=0)
    Ic = _np.take(data_flat, idx_c, axis=0)
    Id = _np.take(data_flat, idx_d, axis=0)
    # (BxHxW)
    xw = bilinear_weights[:, 0]
    yw = bilinear_weights[:, 1]
    # (BxHxW) x 1
    wa = _np.expand_dims((1 - xw) * (1 - yw), 1)
    wb = _np.expand_dims((1 - xw) * yw, 1)
    wc = _np.expand_dims(xw * (1 - yw), 1)
    wd = _np.expand_dims(xw * yw, 1)
    # (BxNP) x D
    resampled_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    # B x NP x D
    return _np.reshape(resampled_flat, batch_shape + [-1, num_feats])


def gradient_image(x):
    x_shape = _ivy.shape(x)
    batch_shape = x_shape[:-3]
    image_dims = x_shape[-3:-1]
    dev_str = _ivy.dev_str(x)
    # to list
    batch_shape = list(batch_shape)
    image_dims = list(image_dims)
    num_dims = x_shape[-1]
    # BS x H-1 x W x D
    dy = x[..., 1:, :, :] - x[..., :-1, :, :]
    # BS x H x W-1 x D
    dx = x[..., :, 1:, :] - x[..., :, :-1, :]
    # BS x H x W x D
    dy = _ivy.concatenate((dy, _ivy.zeros(batch_shape + [1, image_dims[1], num_dims], dev_str=dev_str)), -3)
    dx = _ivy.concatenate((dx, _ivy.zeros(batch_shape + [image_dims[0], 1, num_dims], dev_str=dev_str)), -2)
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
