# local

from .base import FactorizedTensor
import ivy

# global
import warnings


class TRTensor(FactorizedTensor):
    def __init__(self, factors):
        super().__init__()
        shape, rank = TRTensor.validate_tr_tensor(factors)
        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    # Built-ins #
    # ----------#
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
            f"factors list : rank-{self.rank} tensor ring tensor of shape {self.shape}"
        )
        return message

    # Public Methods #
    # ---------------#

    def to_tensor(self):
        return TRTensor.tr_to_tensor(self.factors)

    def to_unfolded(self, mode):
        return TRTensor.tr_to_unfolded(self.factors, mode)

    def to_vec(self):
        return TRTensor.tr_to_vec(self.factors)

    # Properties #
    # ---------------#
    @property
    def n_param(self):
        factors = self.factors
        total_params = sum(int(ivy.prod(tensor.shape)) for tensor in factors)
        return total_params

    # Class Methods #
    # ---------------#
    @staticmethod
    def validate_tr_tensor(factors):
        n_factors = len(factors)

        if n_factors < 2:
            raise ValueError(
                "A Tensor Ring tensor should be composed of at least two factors."
                f"However, {n_factors} factor was given."
            )

        rank = []
        shape = []
        next_rank = None
        for index, factor in enumerate(factors):
            current_rank, current_shape, next_rank = ivy.shape(factor)

            # Check that factors are third order tensors
            if len(factor.shape) != 3:
                raise ValueError(
                    "TR expresses a tensor as third order factors (tr-cores).\n"
                    f"However, ivy.ndim(factors[{index}]) = {len(factor.shape)}"
                )

            # Consecutive factors should have matching ranks
            if ivy.shape(factors[index - 1])[2] != current_rank:
                raise ValueError(
                    "Consecutive factors should have matching ranks\n -- e.g."
                    " ivy.shape(factors[0])[2]) == ivy.shape(factors[1])[0])\nHowever,"
                    f" ivy.shape(factor[{index-1}])[2] =="
                    f" {ivy.shape(factors[index-1])[2]} but"
                    f" ivy.shape(factor[{index}])[0] == {current_rank}"
                )

            shape.append(current_shape)
            rank.append(current_rank)

        # Add last rank (boundary condition)
        rank.append(next_rank)

        return tuple(shape), tuple(rank)

    @staticmethod
    def tr_to_tensor(factors):
        full_shape = [f.shape[1] for f in factors]
        full_tensor = ivy.reshape(factors[0], (-1, factors[0].shape[2]))

        for factor in factors[1:-1]:
            rank_prev, _, rank_next = factor.shape
            factor = ivy.reshape(factor, (rank_prev, -1))
            full_tensor = ivy.dot(full_tensor, factor)
            full_tensor = ivy.reshape(full_tensor, (-1, rank_next))

        full_tensor = ivy.reshape(
            full_tensor, (factors[-1].shape[2], -1, factors[-1].shape[0])
        )
        full_tensor = ivy.moveaxis(full_tensor, 0, -1)
        full_tensor = ivy.reshape(
            full_tensor, (-1, factors[-1].shape[0] * factors[-1].shape[2])
        )
        factor = ivy.moveaxis(factors[-1], -1, 1)
        factor = ivy.reshape(factor, (-1, full_shape[-1]))
        full_tensor = ivy.dot(full_tensor, factor)
        return ivy.reshape(full_tensor, full_shape)

    @staticmethod
    def tr_to_unfolded(factors, mode):
        return ivy.unfold(TRTensor.tr_to_tensor(factors), mode)

    @staticmethod
    def tr_to_vec(factors):
        return ivy.reshape(
            TRTensor.tr_to_tensor(factors),
            (-1,),
        )

    @staticmethod
    def validate_tr_rank(tensor_shape, rank="same", rounding="round"):
        if rounding == "ceil":
            rounding_fun = ivy.ceil
        elif rounding == "floor":
            rounding_fun = ivy.floor
        elif rounding == "round":
            rounding_fun = ivy.round
        else:
            raise ValueError(
                f"Rounding should be round, floor or ceil, but got {rounding}"
            )

        if rank == "same":
            rank = float(1)

        n_dim = len(tensor_shape)
        if n_dim == 2:
            warnings.warn(
                "Determining the TR-rank for the trivial case of a matrix"
                f" (order 2 tensor) of shape {tensor_shape}, not a higher-order tensor."
            )

        if isinstance(rank, float):
            # Choose the *same* rank for each mode
            n_param_tensor = ivy.prod(tensor_shape) * rank

            # R_k I_k R_{k+1} = R^2 I_k
            solution = int(
                rounding_fun(ivy.sqrt(n_param_tensor / ivy.sum(tensor_shape)))
            )
            rank = (solution,) * (n_dim + 1)

        else:
            # Check user input for potential errors
            n_dim = len(tensor_shape)
            if isinstance(rank, int):
                rank = (rank,) * (n_dim + 1)
            elif n_dim + 1 != len(rank):
                message = (
                    "Provided incorrect number of ranks. Should verify len(rank) =="
                    f" len(tensor.shape)+1, but len(rank) = {len(rank)} while"
                    f" len(tensor.shape)+1 = {n_dim + 1}"
                )
                raise ValueError(message)

            # Check first and last rank
            if rank[0] != rank[-1]:
                message = (
                    f"Provided rank[0] == {rank[0]} and rank[-1] == {rank[-1]}"
                    " but boundary conditions dictate rank[0] == rank[-1]"
                )
                raise ValueError(message)

        return list(rank)

    @staticmethod
    def tr_n_param(tensor_shape, rank):
        factor_params = []
        for i, s in enumerate(tensor_shape):
            factor_params.append(rank[i] * s * rank[i + 1])
        return ivy.sum(factor_params)
