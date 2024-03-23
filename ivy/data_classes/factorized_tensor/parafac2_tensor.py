# local
from .base import FactorizedTensor
import ivy

# global
from copy import deepcopy


class Parafac2Tensor(FactorizedTensor):
    def __init__(self, parafac2_tensor):
        super().__init__()

        shape, rank = ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        weights, factors, projections = parafac2_tensor

        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)

        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights
        self.projections = projections

    # Built-ins #
    # ----------#
    def __getitem__(self, index):
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        elif index == 2:
            return self.projections
        else:
            raise IndexError(
                f"You tried to access index {index} of a PARAFAC2 tensor.\n"
                "You can only access index 0, 1 and 2 of a PARAFAC2 tensor"
                "(corresponding respectively to the weights, factors and projections)"
            )

    def __setitem__(self, index, value):
        if index == 0:
            self.weights = value
        elif index == 1:
            self.factors = value
        elif index == 2:
            self.projections = value
        else:
            raise IndexError(
                f"You tried to set index {index} of a PARAFAC2 tensor.\n"
                "You can only set index 0, 1 and 2 of a PARAFAC2 tensor"
                "(corresponding respectively to the weights, factors and projections)"
            )

    def __iter__(self):
        yield self.weights
        yield self.factors
        yield self.projections

    def __len__(self):
        return 3

    def __repr__(self):
        message = (
            f"(weights, factors, projections) : rank-{self.rank} Parafac2Tensor of"
            f" shape {self.shape} "
        )
        return message

    # Public Methods #
    # ---------------#
    def to_tensor(self):
        return ivy.Parafac2Tensor.parafac2_to_tensor(self)

    def to_vec(self):
        return ivy.Parafac2Tensor.parafac2_to_vec(self)

    def to_unfolded(self, mode):
        return ivy.Parafac2Tensor.parafac2_to_unfolded(self, mode)

    # Properties #
    # ---------------#
    @property
    def n_param(self):
        factors_params = self.rank * ivy.sum(self.shape)
        if self.weights:
            return factors_params + self.rank
        else:
            return factors_params

    @classmethod
    def from_CPTensor(cls, cp_tensor, parafac2_tensor_ok=False):
        """Create a Parafac2Tensor from a CPTensor.

        Parameters
        ----------
        cp_tensor
            CPTensor or Parafac2Tensor
            If it is a Parafac2Tensor, then the argument
            ``parafac2_tensor_ok`` must be True'
        parafac2_tensor
            Whether or not Parafac2Tensors can be used as input.

        Returns
        -------
            Parafac2Tensor with factor matrices and weights extracted from a CPTensor
        """
        if parafac2_tensor_ok and len(cp_tensor) == 3:
            return Parafac2Tensor(cp_tensor)
        elif len(cp_tensor) == 3:
            raise TypeError(
                "Input is not a CPTensor. If it is a Parafac2Tensor, then the argument"
                " ``parafac2_tensor_ok`` must be True"
            )

        weights, (A, B, C) = cp_tensor
        Q, R = ivy.qr(B)
        projections = [Q for _ in range(ivy.shape(A)[0])]
        B = R
        return Parafac2Tensor((weights, (A, B, C), projections))

    # Class Methods #
    # ---------------#
    @staticmethod
    def validate_parafac2_tensor(parafac2_tensor):
        """Validate a parafac2_tensor in the form (weights, factors) Return the
        rank and shape of the validated tensor.

        Parameters
        ----------
        parafac2_tensor
            Parafac2Tensor or (weights, factors)

        Returns
        -------
        (shape, rank)
            size of the full tensor and rank of the CP tensor
        """
        if isinstance(parafac2_tensor, ivy.Parafac2Tensor):
            # it's already been validated at creation
            return parafac2_tensor.shape, parafac2_tensor.rank

        weights, factors, projections = parafac2_tensor

        if len(factors) != 3:
            raise ValueError(
                "A PARAFAC2 tensor should be composed of exactly three factors."
                f"However, {len(factors)} factors was given."
            )

        if len(projections) != factors[0].shape[0]:
            raise ValueError(
                "A PARAFAC2 tensor should have one projection matrix for each"
                f" horisontal slice. However, {len(projections)} projection matrices"
                f" was given and the first mode haslength {factors[0].shape[0]}"
            )

        rank = int(ivy.shape(factors[0])[1])

        shape = []
        for i, projection in enumerate(projections):
            current_mode_size, current_rank = ivy.shape(projection)
            if current_rank != rank:
                raise ValueError(
                    "All the projection matrices of a PARAFAC2 tensor should have the"
                    f" same number of columns as the rank. However, rank={rank} but"
                    f" projections[{i}].shape[1]={ivy.shape(projection)[1]}"
                )

            inner_product = ivy.dot(ivy.permute_dims(projection, (1, 0)), projection)
            if (
                ivy.max(
                    ivy.abs(inner_product - ivy.eye(rank, dtype=inner_product[0].dtype))
                )
                > 1e-5
            ):
                raise ValueError(
                    "All the projection matrices must be orthonormal, that is, P.T@P"
                    " = I. "
                    f"However, projection[{i}].T@projection[{i}] -"
                    " T.eye(rank)) = "
                    f"""{ivy.sqrt(ivy.sum(ivy.square(inner_product -
                                  ivy.eye(rank,dtype=inner_product[0].dtype)),
                                    axis=0))}"""
                )

            # Tuple unpacking to possibly support higher
            # order PARAFAC2 tensors in the future
            shape.append((current_mode_size, *[f.shape[0] for f in factors[2:]]))

        # Skip first factor matrix since the rank is extracted from it.
        for i, factor in enumerate(factors[1:]):
            current_mode_size, current_rank = ivy.shape(factor)
            if current_rank != rank:
                raise ValueError(
                    "All the factors of a PARAFAC2 tensor should have the same number"
                    f" of columns.However, factors[0].shape[1]={rank} but"
                    f" factors[{i}].shape[1]={current_rank}."
                )

        if weights is not None and ivy.shape(weights)[0] != rank:
            raise ValueError(
                f"Given factors for a rank-{rank} PARAFAC2 tensor but"
                f" len(weights)={ivy.shape(weights)[0]}."
            )

        return tuple(shape), rank

    @staticmethod
    def parafac2_normalise(parafac2_tensor):
        """Return parafac2_tensor with factors normalised to unit length.

        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
        where the columns of each `V_k` are normalized to unit Euclidean length
        from the columns of `U_k` with the normalizing constants absorbed into
        `weights`. In the special case of a symmetric tensor, `weights` holds the
        eigenvalues of the tensor.

        Parameters
        ----------
        parafac2_tensor
            Parafac2Tensor = (weight, factors, projections)
            factors is list of matrices, all with the same number of columns
            i.e.::
                for u in U:
                    u[i].shape == (s_i, R)

            where `R` is fixed while `s_i` can vary with `i`

        Returns
        -------
        Parafac2Tensor
          normalisation_weights, normalised_factors, normalised_projections
        """
        # allocate variables for weights, and normalized factors
        _, rank = ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        weights, factors, projections = parafac2_tensor

        # if (not copy) and (weights is None):
        #     warnings.warn('Provided copy=False and weights=None: a new Parafac2Tensor'
        #                   'with new weights and factors normalised inplace will
        # be returned.')
        #     weights = T.ones(rank, **T.context(factors[0]))

        # The if test below was added to enable inplace edits
        # however, TensorFlow does not support inplace edits
        # so this is always set to True
        if True:
            factors = [deepcopy(f) for f in factors]
            projections = [deepcopy(p) for p in projections]
            if weights is not None:
                factors[0] = factors[0] * weights
            weights = ivy.ones(rank, dtype=factors[0].dtype)

        for i, factor in enumerate(factors):
            scales = ivy.sqrt(ivy.sum(ivy.square(factor), axis=0))
            weights = weights * scales
            scales_non_zero = ivy.where(
                scales == 0, ivy.ones(ivy.shape(scales), dtype=factors[0].dtype), scales
            )
            factors[i] = factor / scales_non_zero

        return Parafac2Tensor((weights, factors, projections))

    @staticmethod
    def apply_parafac2_projections(parafac2_tensor):
        """Apply the projection matrices to the evolving factor.

        Parameters
        ----------
        parafac2_tensor : Parafac2Tensor

        Returns
        -------
        (weights, factors)
            A tensor decomposition on the form A [B_i] C such that
            the :math:`X_{ijk}` is given by :math:`sum_r A_{ir} [B_i]_{jr} C_{kr}`.

            This is also equivalent to a coupled matrix factorisation, where
            each matrix, :math:`X_i = C diag([a_{i1}, ..., a_{ir}] B_i)`.

            The first element of factors is the A matrix, the second element is
            a list of B-matrices and the third element is the C matrix.
        """
        ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        weights, factors, projections = parafac2_tensor

        evolving_factor = [
            ivy.dot(projection, factors[1]) for projection in projections
        ]

        return weights, (factors[0], evolving_factor, factors[2])

    @staticmethod
    def parafac2_to_slice(parafac2_tensor, slice_idx, validate=True):
        """Generate a single slice along the first mode from the PARAFAC2
        tensor.

        The decomposition is on the form :math:`(A [B_i] C)` such that the
        i-th frontal slice, :math:`X_i`, of :math:`X` is given by

        .. math::

            X_i = B_i diag(a_i) C^T,

        where :math:`diag(a_i)` is the diagonal matrix whose nonzero
        entries are equal to the :math:`i`-th row of the :math:`I times R`
        factor matrix :math:`A`, :math:`B_i`is a :math:`J_i times R` factor
        matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is
        constant for all :math:`i`, and :math:`C` is a :math:`K times R`
        factor matrix. To compute this decomposition, we reformulate
        the expression for :math:`B_i` such that

        .. math::

            B_i = P_i B,

        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`
        is a :math:`R times R` matrix.

        An alternative formulation of the PARAFAC2 decomposition is
        that the tensor element :math:`X_{ijk}` is given by

        .. math::

            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

        with the same constraints hold for :math:`B_i` as above.

        Parameters
        ----------
        parafac2_tensor
             weights
                1D array of shape (rank, ) weights of the factors
            factors
                List of factors of the PARAFAC2 decomposition Contains the
                matrices :math:`A`, :math:`B` and :math:`C` described above
            projection_matrices
                 List of projection matrices used to create evolving factors.

        Returns
        -------
            Full tensor of shape [P[slice_idx].shape[1], C.shape[1]], where
            P is the projection matrices and C is the last factor matrix of
            the Parafac2Tensor.
        """
        if validate:
            ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        weights, (A, B, C), projections = parafac2_tensor
        a = A[slice_idx]
        if weights is not None:
            a = a * weights

        Ct = ivy.permute_dims(C, (1, 0))

        B_i = ivy.dot(projections[slice_idx], B)
        return ivy.dot(B_i * a, Ct)

    @staticmethod
    def parafac2_to_slices(parafac2_tensor, validate=True):
        """Generate all slices along the first mode from a PARAFAC2 tensor.

        Generates a list of all slices from a PARAFAC2 tensor. A list is returned
        since the tensor might have varying size along the second mode. To return
        a tensor, see the ``parafac2_to_tensor`` function instead.shape

        The decomposition is on the form :math:`(A [B_i] C)` such that
        the i-th frontal slice, :math:`X_i`, of :math:`X` is given by

        .. math::

            X_i = B_i diag(a_i) C^T,

        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries are
        equal to the :math:`i`-th row of the :math:`I times R` factor matrix
        :math:`A`, :math:`B_i` is a :math:`J_i times R` factor matrix such
        that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is constant
        for all :math:`i`, and :math:`C` is a :math:`K times R` factor matrix.To
        compute this decomposition, we reformulate the expression for :math:`B_i`
        such that

        .. math::

            B_i = P_i B,

        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`
        is a :math:`R times R` matrix.

        An alternative formulation of the PARAFAC2 decomposition is that the
        tensor element :math:`X_{ijk}` is given by

        .. math::

            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

        with the same constraints hold for :math:`B_i` as above.

        Parameters
        ----------
        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
            * weights : 1D array of shape (rank, )
                weights of the factors
            * factors : List of factors of the PARAFAC2 decomposition
                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
            * projection_matrices : List of projection matrices used to create evolving
                factors.

        Returns
        -------
            A list of full tensors of shapes [P[i].shape[1], C.shape[1]], where
            P is the projection matrices and C is the last factor matrix of the
            Parafac2Tensor.
        """
        if validate:
            ivy.Parafac2Tensor.validate_parafac2_tensor(parafac2_tensor)
        weights, (A, B, C), projections = parafac2_tensor
        if weights is not None:
            A = A * weights
            weights = None

        decomposition = weights, (A, B, C), projections
        I, _ = A.shape  # noqa: E741
        return [
            ivy.Parafac2Tensor.parafac2_to_slice(decomposition, i, validate=False)
            for i in range(I)
        ]

    @staticmethod
    def parafac2_to_tensor(parafac2_tensor):
        """Construct a full tensor from a PARAFAC2 decomposition.

        The decomposition is on the form :math:`(A [B_i] C)` such that the
        i-th frontal slice, :math:`X_i`, of :math:`X` is given by

        .. math::

            X_i = B_i diag(a_i) C^T,

        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries
        are equal to the :math:`i`-th row of the :math:`I times R` factor
        matrix :math:`A`, :math:`B_i` is a :math:`J_i times R` factor matrix
        such that the cross product matrix :math:`B_{i_1}^T B_{i_1}` is
        constant for all :math:`i`, and :math:`C` is a :math:`K times R`
        factor matrix. To compute this decomposition, we reformulate
        the expression for :math:`B_i` such that

        .. math::

            B_i = P_i B,

        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B`
        is a :math:`R times R` matrix.

        An alternative formulation of the PARAFAC2 decomposition is
        that the tensor element :math:`X_{ijk}` is given by

        .. math::

            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

        with the same constraints hold for :math:`B_i` as above.

        Parameters
        ----------
        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
            * weights : 1D array of shape (rank, )
                weights of the factors
            * factors : List of factors of the PARAFAC2 decomposition
                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
            * projection_matrices : List of projection matrices used to create evolving
                factors.

        Returns
        -------
        ndarray
            Full constructed tensor. Uneven slices are padded with zeros.
        """
        _, (A, _, C), projections = parafac2_tensor
        slices = ivy.Parafac2Tensor.parafac2_to_slices(parafac2_tensor)
        lengths = [projection.shape[0] for projection in projections]

        tensor = ivy.zeros(
            (A.shape[0], max(lengths), C.shape[0]), dtype=slices[0].dtype
        )
        for i, (slice_, length) in enumerate(zip(slices, lengths)):
            tensor[i, :length] = slice_
        return tensor

    @staticmethod
    def parafac2_to_unfolded(parafac2_tensor, mode):
        """Construct an unfolded tensor from a PARAFAC2 decomposition. Uneven
        slices are padded by zeros.

        The decomposition is on the form :math:`(A [B_i] C)` such that the
        i-th frontal slice, :math:`X_i`, of :math:`X` is given by

        .. math::

            X_i = B_i diag(a_i) C^T,

        where :math:`diag(a_i)` is the diagonal matrix whose nonzero entries
        are equal to the :math:`i`-th row of the :math:`I times R` factor
        matrix :math:`A`, :math:`B_i` is a :math:`J_i times R` factor
        matrix such that the cross product matrix :math:`B_{i_1}^T B_{i_1}`
        is constant for all :math:`i`, and :math:`C` is a :math:`K times R`
        factor matrix. To compute this decomposition, we reformulate the
        expression for :math:`B_i` such that

        .. math::

            B_i = P_i B,

        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B` is a
        :math:`R times R` matrix.

        An alternative formulation of the PARAFAC2 decomposition is that the
        tensor element :math:`X_{ijk}` is given by

        .. math::

            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

        with the same constraints hold for :math:`B_i` as above.

        Parameters
        ----------
        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
            weights
                weights of the factors
            factors
                Contains the matrices :math:`A`, :math:`B` and :math:`C` described above
            projection_matrices
                factors

        Returns
        -------
            Full constructed tensor. Uneven slices are padded with zeros.
        """
        return ivy.unfold(ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor), mode)

    @staticmethod
    def parafac2_to_vec(parafac2_tensor):
        """Construct a vectorized tensor from a PARAFAC2 decomposition. Uneven
        slices are padded by zeros.

        The decomposition is on the form :math:`(A [B_i] C)` such that
        the i-th frontal slice, :math:`X_i`, of :math:`X` is given by

        .. math::

            X_i = B_i diag(a_i) C^T,

        where :math:`diag(a_i)` is the diagonal matrix whose nonzero
        entries are  equal to the :math:`i`-th row of the :math:`I
        times R` factor matrix :math:`A`, :math:`B_i` is a :math:`J_i
        times R` factor matrix such that the cross product matrix :math:
        `B_{i_1}^T B_{i_1}`is constant for all :math:`i`, and :math:`C`
        is a :math:`K times R` factor matrix. To compute this
        decomposition, we reformulate the expression for :math:`B_i`
        such that

        .. math::

            B_i = P_i B,

        where :math:`P_i` is a :math:`J_i times R` orthogonal matrix and :math:`B` is a
        :math:`R times R` matrix.

        An alternative formulation of the PARAFAC2 decomposition is that
        the tensor element :math:`X_{ijk}` is given by

        .. math::

            X_{ijk} = sum_{r=1}^R A_{ir} B_{ijr} C_{kr},

        with the same constraints hold for :math:`B_i` as above.

        Parameters
        ----------
        parafac2_tensor : Parafac2Tensor - (weight, factors, projection_matrices)
            * weights
            1D array of shape (rank, ) weights of the factors
            * factors
            List of factors of the PARAFAC2 decomposition Contains the matrices
            :math:`A, :math:`B` and :math:`C` described above
            * projection_matrices
                List of projection matrices used to create evolving factors.

        Returns
        -------
            Full constructed tensor. Uneven slices are padded with zeros.6
        """
        return ivy.reshape(ivy.Parafac2Tensor.parafac2_to_tensor(parafac2_tensor), (-1))
