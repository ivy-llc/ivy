from .ivy__ConvNd import ivy__ConvNd
from .ivy__helpers import ivy_conv2d
from .ivy__helpers import ivy_pad


class ivy_Conv2d(ivy__ConvNd):
    def __init__(
        arr,
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

    def _conv_forward(arr, input, weight, bias):
        if arr.padding_mode != "zeros":
            return ivy_conv2d(
                ivy_pad(
                    input, arr._reversed_padding_repeated_twice, mode=arr.padding_mode
                ),
                weight,
                bias,
                arr.stride,
                Translated_parse(0),
                arr.dilation,
                arr.groups,
            )
        return ivy_conv2d(
            input, weight, bias, arr.stride, arr.padding, arr.dilation, arr.groups
        )

    def forward(arr, input):
        return arr._conv_forward(input, arr.weight, arr.bias)
