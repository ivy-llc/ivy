import ivy
from ivy import composition
import ivy.numpy as np
from ivy.framework_handler import current_framework as _cur_framework
from ivy.func_wrapper import (
    with_supported_dtypes,
    with_unsupported_device_and_dtypes,
)
from ..tensor.tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
# Return the brightness-adjusted image
def adjust_brightness(img, brightness_factor):
    # Assuming img is a numpy array or Ivy array
    if ivy.is_array(img):
        # Adjust brightness compositionally using ivy.add and ivy.multiply
        adjusted_img = ivy.add(ivy.multiply(img, brightness_factor), (1 - brightness_factor))
        return adjusted_img
    elif isinstance(img, Image.Image):
        # Convert image to Ivy array
        img_array = np.array(img)
        adjusted_img_array = ivy.add(ivy.multiply(img_array, brightness_factor), (1 - brightness_factor))
        # Convert Ivy array back to image
        adjusted_img_pil = Image.fromarray(adjusted_img_array.astype('uint8'))
        return adjusted_img_pil
    else:
        raise ValueError("Unsupported input format")


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def to_tensor(pic, data_format="CHW"):
    array = ivy.array(pic)
    return Tensor(array)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": ("int8", "uint8", "int16", "float16", "bfloat16", "bool")
        }
    },
    "paddle",
)
@to_ivy_arrays_and_back
def vflip(img, data_format="CHW"):
    if data_format.lower() == "chw":
        axis = -2
    elif data_format.lower() == "hwc":
        axis = -3
    return ivy.flip(img, axis=axis)
