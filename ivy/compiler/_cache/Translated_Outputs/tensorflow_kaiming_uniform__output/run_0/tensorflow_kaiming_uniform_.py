import warnings
import math

from .tensorflow__helpers import tensorflow__calculate_correct_fan
from .tensorflow__helpers import tensorflow_calculate_gain
from .tensorflow__helpers import tensorflow_uniform_


def tensorflow_kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = tensorflow__calculate_correct_fan(tensor, mode)
    gain = tensorflow_calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return tensorflow_uniform_(tensor, -bound, bound, generator=generator)
