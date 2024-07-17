import ivy.functional.frontends.torch.nn.functional as F

from .Translated__ConvNd import Translated__ConvNd
from .helpers import Translated_parse


class Translated_Conv2d(Translated__ConvNd):
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
        kernel_size_ = Translated_parse(kernel_size)
        stride_ = Translated_parse(stride)
        padding_ = padding if isinstance(padding, str) else Translated_parse(padding)
        dilation_ = Translated_parse(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            Translated_parse(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                Translated_parse(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)
