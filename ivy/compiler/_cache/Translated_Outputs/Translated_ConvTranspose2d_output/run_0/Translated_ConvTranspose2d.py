import ivy.functional.frontends.torch as torch

from .Translated__ConvTransposeNd import Translated__ConvTransposeNd
from .helpers import Translated__ntuple_parse

_pair = Translated__ntuple_parse(2, "_pair")


class Translated_ConvTranspose2d(Translated__ConvTransposeNd):
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
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
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

    def forward(self, input, output_size=None):
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
        return torch.nn.functional.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
