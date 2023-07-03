"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
def selu(input_x):
    alpha = 1.67326324
    scale = 1.05070098
    ret = ivy.where(input_x > 0, input_x, alpha * ivy.expm1(input_x))
    arr = scale * ret
    return ivy.astype(arr, input_x.dtype)
