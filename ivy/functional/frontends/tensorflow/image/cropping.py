# global
import functools

# local
import tensorflow as tf
import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.9.0 and below": ("float",)}, "tensorflow")
@to_ivy_arrays_and_back
def extract_patches(images, sizes, strides, rates, padding):
    depth = images.shape[-1]
    kernel_size = functools.reduce(lambda x, y: x * y, sizes, 1)
    kernel_shape = [*sizes[1:-1], depth, kernel_size * depth]
    eye = ivy.eye(kernel_size * depth)
    filters = ivy.reshape(eye, kernel_shape).astype(images.dtype)
    return ivy.conv_general_dilated(
        images,
        filters,
        strides[1:-1],
        padding,
        dilations=rates[1:-1],
    )


@to_ivy_arrays_and_back
def resize(
    image, size, method="bilinear", preserve_aspect_ratio=False, antialias=False
):
    unsqueezed = False
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        unsqueezed = True
    if preserve_aspect_ratio:
        height, width = image.shape[2:]
        new_height, new_width = size
        aspect_ratio = width / height
        new_aspect_ratio = new_width / new_height
        if new_aspect_ratio > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
            new_height = int(new_width * aspect_ratio)
        else:
            new_width = int(new_height / aspect_ratio)
            new_height = int(new_width / aspect_ratio)
    else:
        new_height, new_width = size
    if method == "bicubic":
        method = "bicubic_tensorflow"
    elif method == "area":
        method = "tf_area"
    image = ivy.interpolate(
        image,
        (new_height, new_width),
        mode=method,
        align_corners=False,
        antialias=antialias,
    )
    if unsqueezed:
        return image.squeeze(0)
    return image


@with_supported_dtypes({"2.9.0 and below": ("float",)}, "tensorflow")
@to_ivy_arrays_and_back
def max_pool2d(x, kernel_size, strides=None, padding="VALID"):
    if strides is None:
        strides = kernel_size
    tf_x = ivy.to_numpy(x)
    tf_kernel_size = ivy.to_numpy(kernel_size)
    tf_strides = ivy.to_numpy(strides)
    tf_padding = padding.upper()

    tf_x = tf.transpose(tf_x, [0, 2, 3, 1])  # Convert to NHWC format
    tf_pool = tf.nn.max_pool2d(
        tf_x, ksize=tf_kernel_size, strides=tf_strides, padding=tf_padding
    )
    ivy_pool = ivy.array(tf_pool)
    ivy_pool = ivy.transpose(ivy_pool, [0, 3, 1, 2])  # Convert back to NCHW format
    return ivy_pool
