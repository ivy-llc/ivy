import tensorflow

from .tensorflow__ConvNd import tensorflow__ConvNd
from .tensorflow__helpers import tensorflow_conv2d_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_pad_frnt
from .tensorflow__helpers import tensorflow_parse


class tensorflow_Conv2d(tensorflow__ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        with tensorflow.name_scope("tensorflow_Conv2d/kernel_size_"):
            kernel_size_ = tensorflow_parse(kernel_size)
        with tensorflow.name_scope("tensorflow_Conv2d/stride_"):
            stride_ = tensorflow_parse(stride)
        with tensorflow.name_scope("tensorflow_Conv2d/padding_"):
            padding_ = (
                padding if isinstance(padding, str) else tensorflow_parse(padding)
            )
        with tensorflow.name_scope("tensorflow_Conv2d/dilation_"):
            dilation_ = tensorflow_parse(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            tensorflow_parse(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return tensorflow_conv2d_frnt(
                tensorflow_pad_frnt(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                tensorflow_parse(0),
                self.dilation,
                self.groups,
            )
        return tensorflow_conv2d_frnt(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        return self._conv_forward(input, self.weight, self.bias)
