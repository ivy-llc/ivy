import ivy.functional.frontends.torch.nn as nn

import typing


class Translated__DropoutNd(nn.Module):
    __constants__ = ["p", "inplace"]
    p: typing.Any
    inplace: typing.Any

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        return f"p={self.p}, inplace={self.inplace}"
