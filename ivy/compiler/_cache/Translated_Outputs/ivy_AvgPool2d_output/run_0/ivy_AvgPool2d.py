import typing

from .ivy__AvgPoolNd import ivy__AvgPoolNd
from .ivy__helpers import ivy_avg_pool2d_frnt


class ivy_AvgPool2d(ivy__AvgPoolNd):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ]
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    ceil_mode: typing.Any
    count_include_pad: typing.Any

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        return ivy_avg_pool2d_frnt(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
