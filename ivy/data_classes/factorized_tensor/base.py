import ivy
from collections.abc import Mapping
from abc import ABCMeta


class FactorizedTensor(Mapping, metaclass=ABCMeta):
    """Base Class for Tensors in Factorized form."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tensor(self):
        return NotImplementedError

    def to_unfolded(self, mode):
        return NotImplementedError

    def to_vec(self):
        return NotImplementedError

    def norm(self):
        """Norm l2 of the tensor."""
        return ivy.l2_normalize(self.to_tensor())

    def mode_dot(self, matrix_or_tensor, mode):
        return ivy.mode_dot(self.to_tensor(), matrix_or_tensor, mode)
