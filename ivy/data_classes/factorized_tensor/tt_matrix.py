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

    # Public Methods #
    def to_tensor(self):
        return ivy.TTMatrix.tt_matrix_to_tensor(self)

    def to_matrix(self):
        return ivy.TTMatrix.tt_matrix_to_matrix(self)

    def to_unfolding(self, mode):
        return ivy.TTMatrix.tt_matrix_to_unfolded(self, mode)

    def to_vec(self):
        return ivy.TTMatrix.tt_matrix_to_vec(self)

    # Proprties #
    @property
    def n_param(self):
        n_dim = len(self.shape) // 2
        if n_dim * 2 != len(self.shape):
            msg = (
                "The order of the give tensorized shape is not a multiple of"
                " 2.However, there should be as many dimensions for the left side"
                " (number of rows) as of the right side (number of columns).  For"
                " instance, to convert a matrix of size (8, 9) to the TT-format,  it"
                " can be tensorized to (2, 4, 3, 3) but NOT to (2, 2, 2, 3, 3)."
            )
            raise ValueError(msg)

        left_shape = self.shape[:n_dim]
        right_shape = self.shape[n_dim:]

        factor_params = []
        for i, (ls, rs) in enumerate(zip(left_shape, right_shape)):
            factor_params.append(self.rank[i] * ls * rs * self.rank[i + 1])

        return ivy.sum(factor_params)

    # Class Methods #
    @staticmethod
    def validate_tt_matrix(tt_tensor):
        factors = tt_tensor
        n_factors = len(factors)

        if n_factors < 1:
            raise ValueError(
                "A Tensor-Train (MPS) tensor should be composed of at least one factor."
                f"However, {n_factors} factor was given."
            )

        rank = []
        left_shape = []
        right_shape = []
        for index, factor in enumerate(factors):
            current_rank, current_left_shape, current_right_shape, next_rank = (
                ivy.shape(factor)
            )

            if not len(ivy.shape(factor)) == 4:
                raise ValueError(
                    "A TTMatrix expresses a tensor as fourth order factors"
                    f" (tt-cores).\nHowever, len(ivy.shape(factors[{index}])) ="
                    f" {len(ivy.shape(factor))}"
                )

            if index and ivy.shape(factors[index - 1])[-1] != current_rank:
                raise ValueError(
                    "Consecutive factors should have matching ranks\n -- e.g."
                    " ivy.shape(factors[0])[-1]) =="
                    " ivy.shape(factors[1])[0])\nHowever,"
                    f" ivy.shape(factor[{index-1}])[-1] =="
                    f" {ivy.shape(factors[index - 1])[-1]} but"
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

            left_shape.append(current_left_shape)
            right_shape.append(current_right_shape)

            rank.append(current_rank)

        rank.append(next_rank)
        return tuple(left_shape) + tuple(right_shape), tuple(rank)

    @staticmethod
    def tt_matrix_to_matrix(tt_matrix):
        """Reconstruct the original matrix that was tensorized and compressed
        in the TT- Matrix format.

        Re-assembles 'factors', which represent a tensor in TT-Matrix format
        into the corresponding matrix

        Parameters
        ----------
        factors
            TT-Matrix factors (known as core)
            of shape (rank_k, left_dim_k, right_dim_k, rank_{k+1})

        Returns
        -------
        output_matrix
            matrix whose TT-Matrix decomposition was given by 'factors'
        """
        in_shape = tuple(c.shape[1] for c in tt_matrix)
        return ivy.reshape(
            ivy.TTMatrix.tt_matrix_to_tensor(tt_matrix), (ivy.prod(in_shape), -1)
        )

    @staticmethod
    def tt_matrix_to_tensor(tt_matrix):
        """Return the full tensor whose TT-Matrix decomposition is given by
        'factors'.

        Re-assembles 'factors', which represent a tensor in TT-Matrix format
        into the corresponding full tensor

        Parameters
        ----------
        factors
            TT-Matrix factors (known as core)
            of shape (rank_k, left_dim_k, right_dim_k, rank_{k+1})

        Returns
        -------
        output_tensor
            tensor whose TT-Matrix decomposition was given by 'factors'
        """
        _, in_shape, out_shape, _ = zip(*[ivy.shape(f) for f in tt_matrix])
        ndim = len(in_shape)

        full_shape = sum(zip(*(in_shape, out_shape)), ())
        order = list(range(0, ndim * 2, 2)) + list(range(1, ndim * 2, 2))

        for i, factor in enumerate(tt_matrix):
            if not i:
                res = factor
            else:
                res = ivy.tensordot(res, factor, axes=([-1], [0]))

        # return ivy.transpose(ivy.reshape(res, full_shape), order)
        return ivy.permute_dims(ivy.reshape(res, full_shape), order)

    @staticmethod
    def tt_matrix_to_vec(tt_matrix):
        """Return the tensor defined by its TT-Matrix format ('factors') into
        its vectorized format.

        Parameters
        ----------
        factors
            TT factors

        Returns
        -------
        1-D array
            format of tensor defined by 'factors'
        """
        return ivy.reshape(ivy.TTMatrix.tt_matrix_to_tensor(tt_matrix), (-1,))

    @staticmethod
    def tt_matrix_to_unfolded(tt_matrix, mode):
        """Return the unfolding matrix of a tensor given in TT-Matrix format.

        Reassembles a full tensor from 'factors' and returns its unfolding matrix
        with mode given by 'mode'

        Parameters
        ----------
        factors
            TT-Matrix factors
        mode
            unfolding matrix to be computed along this mode

        Returns
        -------
        2-D array
            unfolding matrix to be computed along this mode
        """
        return ivy.unfold(ivy.TTMatrix.tt_matrix_to_tensor(tt_matrix), mode=mode)

    @staticmethod
    def validate_tt_matrix_rank(tensorized_shape, rank="same"):
        """Return the rank of a TT-Matrix Decomposition.

        Parameters
        ----------
        tensor_shape
            shape of the tensorized matrix to decompose
        rank
            way to determine the rank, by default 'same'
            if 'same': rank is computed to keep the number of
            parameters (at most) the same
            if float, computes a rank so as to keep rank percent of
            the original number of parameters
            if int or tuple, just returns rank
            default is 'same'
        constant_rank
            if True, the *same* rank will be chosen for each modes
            if False (default), the rank of each mode will be proportional to
            the corresponding tensor_shape
            default is False

            *used only if rank == 'same' or 0 < rank <= 1*

        rounding
            Mode for rounding
            One of ["round", "floor", "ceil"]

        Returns
        -------
        rank
            rank of the decomposition
            Tuple of ints
        """
        n_dim = len(tensorized_shape) // 2

        if n_dim * 2 != len(tensorized_shape):
            msg = (
                "The order of the given tensorized shape is not a multiple of"
                " 2. However, there should be as many dimensions for the left side"
                " (number of rows) as of the right side (number of columns).  For"
                " instance, to convert a matrix of size (8, 9) to the TT-format,  it"
                " can be tensorized to (2, 4, 3, 3) but NOT to (2, 2, 2, 3, 3)."
            )
            raise ValueError(msg)

        left_shape = tensorized_shape[:n_dim]
        right_shape = tensorized_shape[n_dim:]

        full_shape = tuple(i * o for i, o in zip(left_shape, right_shape))
        return ivy.TTTensor.validate_tt_rank(full_shape, rank)
