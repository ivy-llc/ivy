from .tensorflow__BatchNorm import tensorflow__BatchNorm
from .tensorflow__helpers import tensorflow_dim


class tensorflow_BatchNorm2d(tensorflow__BatchNorm):
    def _check_input_dim(arr, input):
        if tensorflow_dim(input) != 4:
            raise ValueError(f"expected 4D input (got {tensorflow_dim(input)}D input)")
