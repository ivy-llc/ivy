import ivy.functional.frontends.torch as torch
import ivy.functional.frontends.torch.nn as nn
import ivy.functional.frontends.torch.nn.functional as F

import math
import typing

from .helpers import Translated__calculate_fan_in_and_fan_out
from .helpers import Translated_kaiming_uniform_
from .helpers import Translated_uniform_


class Translated_Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: typing.Any
    out_features: typing.Any
    weight: typing.Any

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        Translated_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = Translated__calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            Translated_uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
