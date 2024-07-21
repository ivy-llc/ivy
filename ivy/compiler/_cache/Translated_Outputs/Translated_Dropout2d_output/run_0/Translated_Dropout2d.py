import ivy.functional.frontends.torch.nn.functional as F

from .Translated__DropoutNd import Translated__DropoutNd


class Translated_Dropout2d(Translated__DropoutNd):
    def forward(self, input):
        return F.dropout2d(input, self.p, self.training, self.inplace)
