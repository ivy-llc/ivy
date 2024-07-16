from .ivy__BatchNorm import ivy__BatchNorm
from .ivy__helpers import ivy_dim


class ivy_BatchNorm2d(ivy__BatchNorm):
    def _check_input_dim(arr, input):
        if ivy_dim(input) != 4:
            raise ValueError(f"expected 4D input (got {ivy_dim(input)}D input)")
