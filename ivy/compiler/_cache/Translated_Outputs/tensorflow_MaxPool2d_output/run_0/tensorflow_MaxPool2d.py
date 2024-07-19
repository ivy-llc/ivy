import typing

from .tensorflow__MaxPoolNd import tensorflow__MaxPoolNd
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_max_pool2d_frnt


class tensorflow_MaxPool2d(tensorflow__MaxPoolNd):
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    dilation: typing.Any

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        return tensorflow_max_pool2d_frnt(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
