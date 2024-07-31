import typing

from .tensorflow__AdaptiveAvgPoolNd import tensorflow__AdaptiveAvgPoolNd
from .tensorflow__helpers import tensorflow_adaptive_avg_pool2d_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output


class tensorflow_AdaptiveAvgPool2d(tensorflow__AdaptiveAvgPoolNd):
    output_size: typing.Any

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        return tensorflow_adaptive_avg_pool2d_frnt(input, self.output_size)
