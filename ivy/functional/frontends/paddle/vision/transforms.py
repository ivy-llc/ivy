import ivy
from ivy.func_wrapper import with_supported_dtypes
from ..tensor.tensor import Tensor
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def to_tensor(pic, data_format="CHW"):
    array = ivy.array(pic)
    return Tensor(array)


@to_ivy_arrays_and_back
def vflip(img, data_format="CHW"):
    if data_format.lower() == "chw":
        axis = -2
    elif data_format.lower() == "hwc":
        axis = -3
    return ivy.flip(img, axis=axis)
