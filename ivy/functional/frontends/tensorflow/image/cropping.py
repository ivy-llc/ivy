# global
import functools

# local
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
