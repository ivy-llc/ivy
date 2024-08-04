from .tensorflow__ConvNd import tensorflow__ConvNd
from .tensorflow__helpers import tensorflow__ntuple
from .tensorflow__helpers import tensorflow_conv2d_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_pad_frnt

_pair = tensorflow__ntuple(2, "_pair")


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
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
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
                _pair(0),
                self.dilation,
                self.groups,
            )
        return tensorflow_conv2d_frnt(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        return self._conv_forward(input, self.weight, self.bias)
