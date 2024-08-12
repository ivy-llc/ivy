import ivy.functional.frontends.torch as torch

from .Translated__DropoutNd import Translated__DropoutNd


class Translated_Dropout2d(Translated__DropoutNd):
    def forward(self, input):
        return torch.nn.functional.dropout2d(input, self.p, self.training, self.inplace)
