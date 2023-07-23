# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
import ivy.functional.frontends.paddle as paddle
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
# Return the brightness-adjusted image
def adjust_brightness(img, brightness_factor):
  return paddle.vision.transforms.adjust_brightness(img, brightness_factor)
