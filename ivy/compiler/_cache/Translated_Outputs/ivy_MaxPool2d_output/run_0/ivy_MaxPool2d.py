import typing

from .ivy__MaxPoolNd import ivy__MaxPoolNd
from .ivy__helpers import ivy_max_pool2d_frnt


class ivy_MaxPool2d(ivy__MaxPoolNd):
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    dilation: typing.Any

    def forward(self, input):
        return ivy_max_pool2d_frnt(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
