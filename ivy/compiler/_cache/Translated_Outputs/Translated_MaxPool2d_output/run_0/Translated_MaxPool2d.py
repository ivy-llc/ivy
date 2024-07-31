import ivy.functional.frontends.torch.nn.functional as F

import typing

from .Translated__MaxPoolNd import Translated__MaxPoolNd


class Translated_MaxPool2d(Translated__MaxPoolNd):
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    dilation: typing.Any

    def forward(self, input):
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
