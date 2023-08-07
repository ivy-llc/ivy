# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from PIL import Image, ImageEnhance, ImageOps

@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
# Return the brightness-adjusted image
def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img
