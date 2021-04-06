"""
Collection of TensorFlow image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
from functools import reduce as _reduce
from operator import mul as _mul
_tfa = None
import tensorflow as _tf

# local
from ivy import tensorflow as _ivy


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
    warp_flat = _tf.reshape(warp, [batch_shape_product] + [-1, 2])
    mat_flat = _tf.reshape(x, [batch_shape_product] + input_image_dims + [-1])
    global _tfa
    if _tfa is None:
        try:
            import tensorflow_addons as _tfa
        except:
            raise Exception('Unable to import tensorflow_addons, verify this is correctly installed.')
    ret = _tfa.image.interpolate_bilinear(mat_flat, warp_flat, indexing='xy')
    return _tf.reshape(ret, batch_shape + [-1, num_feats])


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
