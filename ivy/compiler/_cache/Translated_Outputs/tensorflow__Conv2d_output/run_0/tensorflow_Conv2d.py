from .tensorflow__ConvNd import tensorflow__ConvNd
from .tensorflow__helpers import tensorflow_conv2d
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_pad


class tensorflow_Conv2d(tensorflow__ConvNd):
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
            return tensorflow_conv2d(
                tensorflow_pad(
                    input, arr._reversed_padding_repeated_twice, mode=arr.padding_mode
                ),
                weight,
                bias,
                arr.stride,
                Translated_parse(0),
                arr.dilation,
                arr.groups,
            )
        return tensorflow_conv2d(
            input, weight, bias, arr.stride, arr.padding, arr.dilation, arr.groups
        )

    @tensorflow_handle_transpose_in_input_and_output
    def call(arr, input):
        return arr._conv_forward(input, arr.weight, arr.bias)
