from .base import FactorizedTensor
import ivy

import warnings


class TTTensor(FactorizedTensor):
    def __init__(self, factors, inplace=False):
        super().__init__()

        shape, rank = ivy.TTTensor.validate_tt_tensor(factors)

        self.shape = tuple(shape)
        self.rank = tuple(rank)
        self.factors = factors

    # Built-ins #
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
            f"factors list : rank-{self.rank} matrix-product-state tensor of shape"
            f" {self.shape} "
        )
        return message

    # Public Methods #
    def to_tensor(self):
        return ivy.TTTensor.tt_to_tensor(self)

    def to_unfolding(self, mode):
        return ivy.TTTensor.tt_to_unfolded(self, mode)

    def to_vec(self):
        return ivy.TTTensor.tt_to_vec(self)

    # Properties #
    @property
    def n_param(self):
        factor_params = []
        for i, s in enumerate(self.shape):
            factor_params.append(self.rank[i] * s * self.rank[i + 1])
        return ivy.sum(factor_params)

    # Class Methods #
    @staticmethod
    def validate_tt_tensor(tt_tensor):
        factors = tt_tensor
        n_factors = len(factors)

        if isinstance(tt_tensor, TTTensor):
            return tt_tensor.shape, tt_tensor.rank
        elif isinstance(tt_tensor, (float, int)):
            return 0, 0

        rank = []
        shape = []
        for index, factor in enumerate(factors):
            current_rank, current_shape, next_rank = ivy.shape(factor)

            if len(ivy.shape(factor)) != 3:
                raise ValueError(
                    "TT expresses a tensor as third order factors"
                    f" (tt-cores).\nHowever, len(ivy.shape(factors[{index}])) ="
                    f" {len(ivy.shape(factor))}"
                )

            if index and ivy.shape(factors[index - 1])[2] != current_rank:
                raise ValueError(
                    "Consecutive factors should have matching ranks\n -- e.g."
                    " ivy.shape(factors[0])[2]) == ivy.shape(factors[1])[0])\nHowever,"
                    f" ivy.shape(factor[{index-1}])[2] =="
                    f" {ivy.shape(factors[index - 1])[2]} but"
                    f" ivy.shape(factor[{index}])[0] == {current_rank} "
                )

            if (index == 0) and current_rank != 1:
                raise ValueError(
                    "Boundary conditions dictate factor[0].shape[0] == 1."
                    f"However, got factor[0].shape[0] = {current_rank}."
                )

            if (index == n_factors - 1) and next_rank != 1:
                raise ValueError(
                    "Boundary conditions dictate factor[-1].shape[2] == 1."
                    f"However, got factor[{n_factors}].shape[2] = {next_rank}."
                )

            shape.append(current_shape)
            rank.append(current_rank)

        rank.append(next_rank)

        return tuple(shape), tuple(rank)

    @staticmethod
    def tt_to_tensor(factors):
        """Return the full tensor whose TT decomposition is given by 'factors'.

        Re-assembles 'factors', which represent a tensor in TT/Matrix-Product-State format
        into the corresponding full tensor

        Parameters
        ----------
        factors
            TT factors (TT-cores)

        Returns
        -------
        output_tensor
            tensor whose TT/MPS decomposition was given by 'factors'
        """  # noqa: E501
        if isinstance(factors, (float, int)):
            return factors

        full_shape = [f.shape[1] for f in factors]
        full_tensor = ivy.reshape(factors[0], (full_shape[0], -1))

        for factor in factors[1:]:
            rank_prev, _, rank_next = factor.shape
            factor = ivy.reshape(factor, (rank_prev, -1))
            full_tensor = ivy.matmul(full_tensor, factor)
            full_tensor = ivy.reshape(full_tensor, (-1, rank_next))

        return ivy.reshape(full_tensor, full_shape)

    @staticmethod
    def tt_to_unfolded(factors, mode):
        """Return the unfolding matrix of a tensor given in TT (or Tensor-
        Train) format.

        Reassembles a full tensor from 'factors' and returns its unfolding matrix
        with mode given by 'mode'

        Parameters
        ----------
        factors
            TT factors
        mode
            unfolding matrix to be computed along this mode

        Returns
        -------
        2-D array
        unfolding matrix at mode given by 'mode'
        """
        return ivy.unfold(ivy.TTTensor.tt_to_tensor(factors), mode)

    @staticmethod
    def tt_to_vec(factors):
        """Return the tensor defined by its TT format ('factors') into its
        vectorized format.

        Parameters
        ----------
        factors
            TT factors

        Returns
        -------
        1-D array
        vectorized format of tensor defined by 'factors'
        """
        return ivy.reshape(ivy.TTTensor.tt_to_tensor(factors), (-1,))

    @staticmethod
    def _tt_n_param(tensor_shape, rank):
        """Return the number of parameters of a MPS decomposition for a given
        `rank` and full `tensor_shape`.

        Parameters
        ----------
        tensor_shape
            shape of the full tensor to decompose (or approximate)
        rank
            rank of the MPS decomposition

        Return
        -------
        n_params
            Number of parameters of a MPS decomposition of rank `rank` of
            a full tensor of shape `tensor_shape`
        """
        factor_params = []
        for i, s in enumerate(tensor_shape):
            factor_params.append(rank[i] * s * rank[i + 1])
        return ivy.sum(factor_params)

    @staticmethod
    def validate_tt_rank(
        tensor_shape,
        rank="same",
        constant_rank=False,
        rounding="round",
        allow_overparametrization=True,
    ):
        """Return the rank of a TT Decomposition.

        Parameters
        ----------
        tensor_shape
            shape of the tensor to decompose
        rank
            way to determine the rank, by default 'same'
            if 'same': rank is computed to keep the number of parameters (at most) the same
            if float, computes a rank so as to keep rank percent of the original number of parameters
            if int or tuple, just returns rank
        constant_rank
            if True, the *same* rank will be chosen for each modes
            if False (default), the rank of each mode will be
            proportional to the corresponding tensor_shape
            used only if rank == 'same' or 0 < rank <= 1*

        rounding
            Mode for rounding
            One of ["round", "floor", "ceil"]

        allow_overparametrization
            if False, the rank must be realizable through iterative application of SVD

        Returns
        -------
        rank
            rank of the decomposition
        """  # noqa: E501
        if rounding == "ceil":
            rounding_fn = ivy.ceil
        elif rounding == "floor":
            rounding_fn = ivy.floor
        elif rounding == "round":
            rounding_fn = ivy.round
        else:
            raise ValueError(
                f"Rounding should be round, floor or ceil, but got {rounding}"
            )

        if rank == "same":
            rank = float(1)

        if isinstance(rank, float) and constant_rank:
            n_param_tensor = ivy.prod(tensor_shape) * rank
            order = len(tensor_shape)

            if order == 2:
                rank = (1, n_param_tensor / (tensor_shape[0] + tensor_shape[1]), 1)
                warnings.warn(
                    "Determining the tt-rank for the trivial case of a matrix (order 2"
                    f" tensor) of shape {tensor_shape}, not a higher-order tensor."
                )

            a = ivy.sum(tensor_shape[1:-1])
            b = ivy.sum(tensor_shape[0] + tensor_shape[-1])
            c = -n_param_tensor
            delta = ivy.sqrt(b**2 - 4 * a * c)
            solution = int(rounding_fn((-b + delta) / (2 * a)))
            rank = rank = (1,) + (solution,) * (order - 1) + (1,)

        elif isinstance(rank, float):
            order = len(tensor_shape)
            avg_dim = [
                (tensor_shape[i] + tensor_shape[i + 1]) / 2 for i in range(order - 1)
            ]
            if len(avg_dim) > 1:
                a = sum(
                    avg_dim[i - 1] * tensor_shape[i] * avg_dim[i]
                    for i in range(1, order - 1)
                )
            else:
                warnings.warn(
                    "Determining the tt-rank for the trivial case of a matrix (order 2"
                    f" tensor) of shape {tensor_shape}, not a higher-order tensor."
                )
                a = avg_dim[0] ** 2 * tensor_shape[0]
            b = tensor_shape[0] * avg_dim[0] + tensor_shape[-1] * avg_dim[-1]
            c = -ivy.prod(tensor_shape) * rank
            delta = ivy.sqrt(b**2 - 4 * a * c)

            fraction_param = (-b + delta) / (2 * a)
            rank = tuple(max(int(rounding_fn(d * fraction_param)), 1) for d in avg_dim)
            rank = (1,) + rank + (1,)

        else:
            n_dim = len(tensor_shape)
            if isinstance(rank, int):
                rank = [1] + [rank] * (n_dim - 1) + [1]
            elif n_dim + 1 != len(rank):
                message = (
                    "Provided incorrect number of ranks. Should verify len(rank) =="
                    f" len(ivy.shape(tensor)) + 1, but len(rank) = {len(rank)} while"
                    f" len(ivy.shape(tensor)) + 1  = {n_dim+1}"
                )
                raise (ValueError(message))

            if rank[0] != 1:
                message = (
                    f"Provided rank[0] == {rank[0]} but boundary conditions dictate"
                    " rank[0] == rank[-1] == 1."
                )
                raise ValueError(message)
            if rank[-1] != 1:
                message = (
                    f"Provided rank[-1] == {rank[-1]} but boundary conditions dictate"
                    " rank[0] == rank[-1] == 1."
                )
                raise ValueError(message)

        if allow_overparametrization:
            return list(rank)
        else:
            validated_rank = [1]
            for i, s in enumerate(tensor_shape[:-1]):
                n_row = int(rank[i] * s)
                n_column = ivy.prod(tensor_shape[(i + 1) :])
                validated_rank.append(min(n_row, n_column, rank[i + 1]))
            validated_rank.append(1)

            return validated_rank

    @staticmethod
    def pad_tt_rank(factor_list, n_padding=1, pad_boundaries=False):
        """Pad the factors of a Tensor-Train so as to increase its rank without
        changing its reconstruction.

        The tensor-train (ring) will be padded with 0s to increase its rank only but
        not the underlying tensor it represents.

        Parameters
        ----------
        factor_list
            tensor list
        n_padding
            how much to increase the rank (bond dimension) by
        pad_boundaries
            if True, also pad the boundaries (useful for a tensor-ring)
            should be False for a tensor-train to keep the boundary rank to be 1

        Returns
        -------
        padded_factor_list
        """
        new_factors = []
        n_factors = len(factor_list)

        for i, factor in enumerate(factor_list):
            n_padding_left = n_padding_right = n_padding
            if (i == 0) and not pad_boundaries:
                n_padding_left = 0
            elif (i == n_factors - 1) and not pad_boundaries:
                n_padding_right = 0

            r1, *s, r2 = ivy.shape(factor)
            new_factor = ivy.zeros((r1 + n_padding_left, *s, r2 + n_padding_right))
            new_factors.append(
                ivy.TTTensor.index_update(
                    new_factor,
                    (slice(None, r1, None), ..., slice(None, r2, None)),
                    factor,
                )
            )

        return new_factors

    @staticmethod
    def index_update(tensor, indices, values):
        tensor[indices] = values
        return tensor
