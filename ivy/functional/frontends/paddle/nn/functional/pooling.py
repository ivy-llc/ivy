# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
def avg_pool1d(
    x, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None
):
    data_format = "NCL"
    exclusive = not exclusive

    return ivy.avg_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        count_include_pad=exclusive,
        ceil_mode=ceil_mode,
        data_format=data_format,
    )
