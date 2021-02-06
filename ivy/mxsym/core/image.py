"""
Collection of MXNet image functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math

# local
from ivy import mxsym as _ivy


def stack_images(images, desired_aspect_ratio=(1, 1), batch_shape=None, image_dims=None):
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
    stack_width_int = math.ceil(num_images/stack_height)
    image_rows = list()
    for i in range(stack_width_int):
        images_to_concat = images[i*stack_height_int:(i+1)*stack_height_int]
        images_to_concat += [_ivy.zeros_like(images[0])] * (stack_height_int - len(images_to_concat))
        image_rows.append(_ivy.concatenate(images_to_concat, num_batch_dims))
    return _ivy.concatenate(image_rows, num_batch_dims + 1)


def bilinear_resample(*_):
    raise Exception('mxnet symbolic does not support ivy.image.bilinear_resample(),'
                    'as there is no BilinearSampler class')


def gradient_image(*_):
    raise Exception('mxnet symbolic does not support ivy.image.gradient_image(),'
                    'as array slicing is not possible in symbolic mode.')
