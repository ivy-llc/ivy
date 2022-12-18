import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
)


@to_ivy_arrays_and_back
def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return ivy.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
    )
