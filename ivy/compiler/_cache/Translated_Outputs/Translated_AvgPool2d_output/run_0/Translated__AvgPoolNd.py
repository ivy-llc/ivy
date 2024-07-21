import ivy.functional.frontends.torch.nn as nn


class Translated__AvgPoolNd(nn.Module):
    __constants__ = [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
    ]

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
