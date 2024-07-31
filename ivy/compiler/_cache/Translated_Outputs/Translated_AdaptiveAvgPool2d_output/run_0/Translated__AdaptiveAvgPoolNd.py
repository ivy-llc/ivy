import ivy.functional.frontends.torch.nn as nn


class Translated__AdaptiveAvgPoolNd(nn.Module):
    __constants__ = ["output_size"]

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def extra_repr(self):
        return f"output_size={self.output_size}"
