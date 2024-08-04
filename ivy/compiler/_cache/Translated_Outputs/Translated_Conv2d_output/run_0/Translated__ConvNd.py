import ivy.functional.frontends.torch as torch
import ivy.functional.frontends.torch.nn as nn

import math
import typing
from typing import Optional

from .helpers import Translated__calculate_fan_in_and_fan_out
from .helpers import Translated__reverse_repeat_tuple
from .helpers import Translated_kaiming_uniform_
from .helpers import Translated_uniform_


class Translated__ConvNd(nn.Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(self, input, weight, bias): ...

    in_channels: typing.Any
    _reversed_padding_repeated_twice: typing.Any
    out_channels: typing.Any
    kernel_size: typing.Any
    stride: typing.Any
    padding: typing.Any
    dilation: typing.Any
    transposed: typing.Any
    output_padding: typing.Any
    groups: typing.Any
    padding_mode: typing.Any
    weight: typing.Any
    bias: typing.Any

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = Translated__reverse_repeat_tuple(
                self.padding, 2
            )
        if transposed:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_channels, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        Translated_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = Translated__calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                Translated_uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
