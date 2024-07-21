from .ivy__DropoutNd import ivy__DropoutNd
from .ivy__helpers import ivy_dropout2d_frnt


class ivy_Dropout2d(ivy__DropoutNd):
    def forward(self, input):
        return ivy_dropout2d_frnt(input, self.p, self.training, self.inplace)
