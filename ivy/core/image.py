"""
Collection of image Ivy functions.
"""

# local
import ivy as _ivy
import numpy as _np
from ivy.framework_handler import current_framework as _cur_framework


def stack_images(images, desired_aspect_ratio=(1, 1), f=None):
    """
    Stacks a group of images into a combined windowed image, fitting the desired aspect ratio as closely as possible.

    :param images: Sequence of image arrays to be stacked *[batch_shape,height,width,dims]* .
    :type images: sequence of arrays
    :param desired_aspect_ratio: Desired aspect ratio of stacked image.
    :type desired_aspect_ratio: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Stacked image, suitable for viewing in a single window.
    """
    return _cur_framework(images[0], f=f).stack_images(images, desired_aspect_ratio)


def bilinear_resample(x, warp, f=None):
    """
    Performs bilinearly re-sampling on input image.

    :param x: Input image *[batch_shape,h,w,dims]*.
    :type x: array
    :param warp: Warp array *[batch_shape,num_samples,2]*
    :type warp: array
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Image after bilinear re-sampling.
    """
    return _cur_framework(x, f=f).bilinear_resample(x, warp)


def gradient_image(x, f=None):
    """
    Computes image gradients (dy, dx) for each channel.

    :param x: Input image *[batch_shape, h, w, d]* .
    :type x: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .
    """
    return _cur_framework(x, f=f).gradient_image(x)


def float_img_to_uint8_img(x, f=None):
    """
    Converts an image of floats into a bit-cast 4-channel image of uint8s, which can be saved to disk.

    :param x: Input float image *[batch_shape,h,w]*.
    :type x: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The new encoded uint8 image *[batch_shape,h,w,4]* .
    """
    x_np = _ivy.to_numpy(x)
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_uint8 = _np.frombuffer(x_bytes, _np.uint8)
    return _ivy.array(_np.reshape(x_uint8, list(x_shape) + [4]).tolist())


def uint8_img_to_float_img(x, f=None):
    """
    Converts an image of uint8 values into a bit-cast float image.

    :param x: Input uint8 image *[batch_shape,h,w,4]*.
    :type x: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: The new float image *[batch_shape,h,w]* .
    """
    x_np = _ivy.to_numpy(x)
    x_shape = x_np.shape
    x_bytes = x_np.tobytes()
    x_float = _np.frombuffer(x_bytes, _np.float32)
    return _ivy.array(_np.reshape(x_float, x_shape[:-1]).tolist())
