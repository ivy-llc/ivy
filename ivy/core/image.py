"""
Collection of image Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


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
    return _get_framework(images[0], f=f).stack_images(images, desired_aspect_ratio)


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
    return _get_framework(x, f=f).bilinear_resample(x, warp)


def gradient_image(x, f=None):
    """
    Computes image gradients (dy, dx) for each channel.

    :param x: Input image *[batch_shape, h, w, d]* .
    :type x: array
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .
    """
    return _get_framework(x, f=f).gradient_image(x)
