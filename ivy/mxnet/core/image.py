"""
Collection of MXNet image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
from functools import reduce as _reduce
from operator import mul as _mul
import mxnet as _mx

# local
from ivy import mxnet as _ivy


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


def bilinear_resample(x, warp):
    batch_shape = _ivy.shape(x)[:-3]
    input_image_dims = _ivy.shape(x)[-3:-1]
    num_feats = x.shape[-1]
    batch_shape = list(batch_shape)
    input_image_dims = list(input_image_dims)
    batch_shape_product = _reduce(_mul, batch_shape, 1)
    warp_flat = _mx.nd.reshape(warp, [batch_shape_product] + [-1, 1] + [2])
    warp_flat_x = 2 * warp_flat[..., 0:1] / (input_image_dims[1] - 1) - 1
    warp_flat_y = 2 * warp_flat[..., 1:2] / (input_image_dims[0] - 1) - 1
    warp_flat_scaled = _mx.nd.concat(warp_flat_x, warp_flat_y, dim=-1)
    warp_flat_trans = _mx.nd.transpose(warp_flat_scaled, (0, 3, 1, 2))
    mat_flat = _mx.nd.reshape(x, [batch_shape_product] + input_image_dims + [-1])
    mat_flat_trans = _mx.nd.transpose(mat_flat, (0, 3, 1, 2))
    interpolated_flat_transposed = _mx.nd.BilinearSampler(mat_flat_trans, warp_flat_trans)
    interpolated_flat = _mx.nd.transpose(interpolated_flat_transposed, (0, 2, 3, 1))
    return _mx.nd.reshape(interpolated_flat, batch_shape + [-1, num_feats])


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
    dy = _ivy.concatenate((dy, _mx.nd.zeros(batch_shape + [1, image_dims[1], num_dims], ctx=ctx)), -3)
    # noinspection PyTypeChecker
    dx = _ivy.concatenate((dx, _mx.nd.zeros(batch_shape + [image_dims[0], 1, num_dims], ctx=ctx)), -2)
    # BS x H x W x D,    BS x H x W x D
    return dy, dx
