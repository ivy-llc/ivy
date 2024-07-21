import ivy.functional.frontends.torch.nn.functional as F

import typing

from .Translated__AdaptiveAvgPoolNd import Translated__AdaptiveAvgPoolNd


class Translated_AdaptiveAvgPool2d(Translated__AdaptiveAvgPoolNd):
    output_size: typing.Any

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)
