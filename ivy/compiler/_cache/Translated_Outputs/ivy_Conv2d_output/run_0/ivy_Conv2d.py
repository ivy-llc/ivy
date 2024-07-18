from .ivy__ConvNd import ivy__ConvNd
from .ivy__helpers import ivy_conv2d
from .ivy__helpers import ivy_pad
from .ivy__helpers import ivy_parse


class ivy_Conv2d(ivy__ConvNd):
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
        kernel_size_ = ivy_parse(kernel_size)
        stride_ = ivy_parse(stride)
        padding_ = padding if isinstance(padding, str) else ivy_parse(padding)
        dilation_ = ivy_parse(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            ivy_parse(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return ivy_conv2d(
                ivy_pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                ivy_parse(0),
                self.dilation,
                self.groups,
            )
        return ivy_conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)
