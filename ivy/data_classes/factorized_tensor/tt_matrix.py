from .base import FactorizedTensor
import ivy


class TTMatrix(FactorizedTensor):
    def __init__(self, factors, implace=False):
        super().__init__()

        shape, rank = ivy.TTMatrix.validate_tt_matrix(factors)

        self.shape = tuple(shape)
        self.order = len(self.shape) // 2
        self.left_shape = self.shape[: self.order]
        self.right_shape = self.shape[self.order :]
        self.rank = tuple(rank)
        self.factors = factors

    def __getitem__(self, index):
        return self.factors[index]

    def __setitem__(self, index, value):
        self.factors[index] = value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __len__(self):
        return len(self.factors)

    def __repr__(self):
        message = (
            f"factors list : rank-{self.rank} TT-Matrix of tensorized shape"
            f" {self.shape} corresponding to a matrix of size"
            f" {ivy.prod(self.left_shape)} x {ivy.prod(self.right_shape)}"
        )
        return message

    def to_tensor(self):
        return ivy.TTMatrix.tt_matrix_to_tensor(self)

    def to_matrix(self):
        return ivy.TTMatrix.tt_matrix_to_matrix(self)

    def to_unfolding(self, mode):
        return ivy.TTMatrix.tt_matrix_to_unfolded(self, mode)

    def to_vec(self):
        return ivy.TTMatrix.tt_matrix_to_vec(self)
