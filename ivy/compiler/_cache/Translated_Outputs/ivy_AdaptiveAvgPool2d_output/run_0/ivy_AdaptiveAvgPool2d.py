import typing

from .ivy__AdaptiveAvgPoolNd import ivy__AdaptiveAvgPoolNd
from .ivy__helpers import ivy_adaptive_avg_pool2d_frnt


class ivy_AdaptiveAvgPool2d(ivy__AdaptiveAvgPoolNd):
    output_size: typing.Any

    def forward(self, input):
        return ivy_adaptive_avg_pool2d_frnt(input, self.output_size)
