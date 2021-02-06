"""
Collection of image Ivy functions.
"""

# local
from ivy.framework_handler import get_framework as _get_framework


def stack_images(images, desired_aspect_ratio=(1, 1), batch_shape=None, image_dims=None, f=None):
    """
    Stacks a group of images into a combined windowed image, fitting the desired aspect ratio as closely as possible.

    :param images: Sequence of image arrays to be stacked *[batch_shape,height,width,dims]* .
    :type images: sequence of arrays
    :param desired_aspect_ratio: Desired aspect ratio of stacked image.
    :type desired_aspect_ratio: sequence of ints
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: image dimensions.
    :type image_dims: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Stacked image, suitable for viewing in a single window.
    """
    return _get_framework(images[0], f=f).stack_images(images, desired_aspect_ratio, batch_shape, image_dims)


def bilinear_resample(x, warp, batch_shape=None, input_image_dims=None, output_image_dims=None, f=None):
    """
    Performs bilinearly re-sampling on input image.

    :param x: Input image.
    :type x: array
    :param warp: Warp array.
    :type warp: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param input_image_dims: image dimensions of image to be sampled from.
    :type input_image_dims: sequence of ints
    :param output_image_dims: image dimensions of the output image, after sampling.
    :type output_image_dims: sequence of ints
    :param f: Machine learning framework. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Image after bilinear re-sampling.
    """
    return _get_framework(x, f=f).bilinear_resample(x, warp, batch_shape, input_image_dims, output_image_dims)


def gradient_image(x, batch_shape=None, image_dims=None, dev=None, f=None):
    """
    Computes image gradients (dy, dx) for each channel.

    :param x: Input image *[batch_shape, h, w, d]* .
    :type x: array
    :param batch_shape: Shape of batch. Inferred from inputs if None.
    :type batch_shape: sequence of ints, optional
    :param image_dims: Image dimensions. Inferred from inputs in None.
    :type image_dims: sequence of ints, optional
    :param dev: device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if None.
    :type dev: str, optional
    :param f: Machine learning library. Inferred from inputs if None.
    :type f: ml_framework, optional
    :return: Gradient images dy *[batch_shape,h,w,d]* and dx *[batch_shape,h,w,d]* .
    """
    return _get_framework(x, f=f).gradient_image(x, batch_shape, image_dims, dev)
