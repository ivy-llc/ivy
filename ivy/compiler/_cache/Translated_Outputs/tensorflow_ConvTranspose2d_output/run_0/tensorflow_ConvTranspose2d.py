import tensorflow

from .tensorflow__ConvTransposeNd import tensorflow__ConvTransposeNd
from .tensorflow__helpers import tensorflow_conv_transpose2d_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_parse


class tensorflow_ConvTranspose2d(tensorflow__ConvTransposeNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        with tensorflow.name_scope("tensorflow_ConvTranspose2d/kernel_size"):
            kernel_size = tensorflow_parse(kernel_size)
        with tensorflow.name_scope("tensorflow_ConvTranspose2d/stride"):
            stride = tensorflow_parse(stride)
        with tensorflow.name_scope("tensorflow_ConvTranspose2d/padding"):
            padding = tensorflow_parse(padding)
        with tensorflow.name_scope("tensorflow_ConvTranspose2d/dilation"):
            dilation = tensorflow_parse(dilation)
        with tensorflow.name_scope("tensorflow_ConvTranspose2d/output_padding"):
            output_padding = tensorflow_parse(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )
        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        return tensorflow_conv_transpose2d_frnt(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
