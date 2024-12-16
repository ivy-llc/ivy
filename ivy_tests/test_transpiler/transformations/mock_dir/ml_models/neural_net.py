import numpy as np
from ..data_utils.preprocessing import Preprocessor
import torch.nn as nn
from torch.nn.functional import pad, normalize


class NeuralNetwork:
    def __init__(self):
        self.layer = nn.Conv2d(64, 256, 32)

    def forward(self, x):
        inp = Preprocessor.scale(np.sum(x))
        padded_inp = pad(inp)
        logits = self.layer(padded_inp)
        return normalize(logits)


MODEL_VERSION = "1.0.0"
