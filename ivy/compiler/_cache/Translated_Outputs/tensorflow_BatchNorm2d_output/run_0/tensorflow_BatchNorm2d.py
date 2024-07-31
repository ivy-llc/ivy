from .tensorflow__BatchNorm import tensorflow__BatchNorm
from .tensorflow__helpers import tensorflow_dim_frnt_


class tensorflow_BatchNorm2d(tensorflow__BatchNorm):
    def _check_input_dim(self, input):
        if tensorflow_dim_frnt_(input) != 4:
            raise ValueError(
                f"expected 4D input (got {tensorflow_dim_frnt_(input)}D input)"
            )
