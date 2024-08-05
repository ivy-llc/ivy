from .ivy__ConvNd import ivy__ConvNd
from .ivy__helpers import ivy__ntuple_parse
from .ivy__helpers import ivy_conv2d_frnt
from .ivy__helpers import ivy_pad_frnt

_pair = ivy__ntuple_parse(2, "_pair")


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
            return ivy_conv2d_frnt(
                ivy_pad_frnt(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return ivy_conv2d_frnt(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)
