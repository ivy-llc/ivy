# local
from .base import FactorizedTensor
import ivy


class CPTensor(FactorizedTensor):
    def __init__(self, cp_tensor):
        super().__init__()

        shape, rank = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor

        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)

        self.shape = shape
        self.rank = rank
        self.factors = factors
        self.weights = weights

    # Built-ins #
    def __getitem__(self, index):
        if index == 0:
            return self.weights
        elif index == 1:
            return self.factors
        else:
            raise IndexError(
                f"You tried to access index {index} of a CP tensor.\n"
                "You can only access index 0 and 1 of a CP tensor"
                "(corresponding respectively to the weights and factors)"
            )

    def __setitem__(self, index, value):
        if index == 0:
            self.weights = value
        elif index == 1:
            self.factors = value
        else:
            raise IndexError(
                f"You tried to set the value at index {index} of a CP tensor.\n"
                "You can only set index 0 and 1 of a CP tensor"
                "(corresponding respectively to the weights and factors)"
            )

    def __iter__(self):
        yield self.weights
        yield self.factors

    def __len__(self):
        return 2

    def __repr__(self):
        message = (
            f"(weights, factors) : rank-{self.rank} CPTensor of shape {self.shape}"
        )
        return message

    # Public Methods #
    # ---------------#
    def to_tensor(self):
        return ivy.CPTensor.cp_to_tensor(self)

    def to_vec(self):
        return ivy.CPTensor.cp_to_vec(self)

    def to_unfolded(self, mode):
        return ivy.CPTensor.cp_to_unfolded(self, mode)

    def cp_copy(self):
        return CPTensor(
            (
                ivy.copy_array(self.weights),
                [ivy.copy_array(self.factors[i]) for i in range(len(self.factors))],
            )
        )

    def mode_dot(self, matrix_or_vector, mode, keep_dim=False, copy=True):
        """
        N-mode product of a CP tensor and a matrix or vector at the specified mode.

        Parameters
        ----------
        cp_tensor : tl.CPTensor or (core, factors)

        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        CPTensor = (core, factors)
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
              if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`
              if matrix_or_vector is a vector

        See Also
        --------
        cp_mode_dot : chaining several mode_dot in one call
        """
        return ivy.CPTensor.cp_mode_dot(
            self, matrix_or_vector, mode, keep_dim=keep_dim, copy=copy
        )

    def norm(self):
        """
        Return the l2 norm of a CP tensor.

        Parameters
        ----------
        cp_tensor : tl.CPTensor or (core, factors)

        Returns
        -------
        l2-norm : int

        Notes
        -----
        This is ||cp_to_tensor(factors)||^2

        You can see this using the fact that
        khatria-rao(A, B)^T x khatri-rao(A, B) = A^T x A  * B^T x B
        """
        return ivy.CPTensor.cp_norm(self)

    def normalize(self, inplace=True):
        """
        Normalize the factors to unit length.

        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
        where the columns of each `V_k` are normalized to unit Euclidean length
        from the columns of `U_k` with the normalizing constants absorbed into
        `weights`. In the special case of a symmetric tensor, `weights` holds the
        eigenvalues of the tensor.

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is list of matrices, all with the same number of columns
            i.e.::
                for u in U:
                    u[i].shape == (s_i, R)

            where `R` is fixed while `s_i` can vary with `i`

        inplace : bool, default is True
            if False, returns a normalized Copy
            otherwise the tensor modifies itself and returns itself

        Returns
        -------
        CPTensor = (normalisation_weights, normalised_factors)
            returns itself if inplace is False, a normalized copy otherwise
        """
        self.weights, self.factors = ivy.CPTensor.cp_normalize(self)

    # Properties #
    # ---------------#
    @property
    def n_param(self):
        factors_params = self.rank * ivy.sum(self.shape)
        if self.weights:
            return factors_params + self.rank
        else:
            return factors_params

    # Class Methods #
    # ---------------#
    @staticmethod
    def validate_cp_tensor(cp_tensor):
        """
        Validate a cp_tensor in the form (weights, factors)

            Return the rank and shape of the validated tensor

        Parameters
        ----------
        cp_tensor : CPTensor or (weights, factors)

        Returns
        -------
        (shape, rank) : (int tuple, int)
            size of the full tensor and rank of the CP tensor
        """
        if isinstance(cp_tensor, CPTensor):
            # it's already been validated at creation
            return cp_tensor.shape, cp_tensor.rank
        elif isinstance(cp_tensor, (float, int)):  # 0-order tensor
            return 0, 0

        weights, factors = cp_tensor

        ndim = len(factors[0].shape)
        if ndim == 2:
            rank = int(ivy.shape(factors[0])[1])
        elif ndim == 1:
            rank = 1
        else:
            raise ValueError(
                "Got a factor with 3 dimensions but CP factors should be at most 2D, of"
                " shape (size, rank)."
            )

        shape = []
        for i, factor in enumerate(factors):
            s = ivy.shape(factor)
            if len(s) == 2:
                current_mode_size, current_rank = s
            else:  # The shape is just (size, ) if rank 1
                current_mode_size, current_rank = *s, 1

            if current_rank != rank:
                raise ValueError(
                    "All the factors of a CP tensor should have the same number of"
                    f" column.However, factors[0].shape[1]={rank} but"
                    f" factors[{i}].shape[1]={ivy.shape(factor)[1]}."
                )
            shape.append(current_mode_size)

        if weights is not None and ivy.shape(weights) != (rank,):
            raise ValueError(
                f"Given factors for a rank-{rank} CP tensor but"
                f" len(weights)={ivy.shape(weights)}."
            )

        return tuple(shape), rank

    @staticmethod
    def cp_n_param(tensor_shape, rank, weights=False):
        """
        Return number of parameters of a CP decomposition for a given `rank` and full
        `tensor_shape`.

        Parameters
        ----------
        tensor_shape
            shape of the full tensor to decompose (or approximate)

        rank
            rank of the CP decomposition

        Returns
        -------
        n_params : int
            Number of parameters of a CP decomposition of rank `rank`
              of a full tensor of shape `tensor_shape`
        """
        factors_params = rank * ivy.sum(tensor_shape)
        if weights:
            return factors_params + rank
        else:
            return factors_params

    @staticmethod
    def validate_cp_rank(tensor_shape, rank="same", rounding="round"):
        """
        Return the rank of a CP Decomposition.

        Parameters
        ----------
        tensor_shape
            shape of the tensor to decompose
        rank : {'same', float, int}, default is same
            way to determine the rank, by default 'same'
            if 'same': rank is computed to keep the number
            of parameters (at most) the same
            if float, computes a rank so as to keep rank
            percent of the original number of parameters
            if int, just returns rank
        rounding = {'round', 'floor', 'ceil'}

        Returns
        -------
        rank : int
            rank of the decomposition
        """
        if rounding == "ceil":
            rounding_fun = ivy.ceil
        elif rounding == "floor":
            rounding_fun = ivy.floor
        elif rounding == "round":
            rounding_fun = ivy.round
        else:
            raise ValueError(
                f"Rounding should be of round, floor or ceil, but got {rounding}"
            )

        if rank == "same":
            rank = float(1)

        if isinstance(rank, float):
            rank = int(
                rounding_fun(ivy.prod(tensor_shape) * rank / ivy.sum(tensor_shape))
            )
        return rank

    @staticmethod
    def cp_normalize(cp_tensor):
        """
        Return cp_tensor with factors normalised to unit length.

        Turns ``factors = [|U_1, ... U_n|]`` into ``[weights;
        |V_1, ... V_n|]``, where the columns of each `V_k` are
        normalized to unit Euclidean length from the columns of
        `U_k` with the normalizing constants absorbed into
        `weights`. In the special case of a symmetric tensor,
        `weights` holds the eigenvalues of the tensor.

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is list of matrices,
              all with the same number of columns
            i.e.::

                for u in U:
                    u[i].shape == (s_i, R)

            where `R` is fixed while `s_i` can vary with `i`

        Returns
        -------
        CPTensor = (normalisation_weights, normalised_factors)
        """
        _, rank = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor

        if weights is None:
            weights = ivy.ones(rank, dtype=factors[0].dtype)

        normalized_factors = []
        for i, factor in enumerate(factors):
            if i == 0:
                factor = factor * weights
                weights = ivy.ones((rank,), dtype=factor.dtype)

            scales = ivy.sqrt(ivy.sum(ivy.square(factor), axis=0))
            scales_non_zero = ivy.where(
                scales == 0, ivy.ones(ivy.shape(scales), dtype=factor.dtype), scales
            )
            weights = weights * scales
            normalized_factors.append(factor / ivy.reshape(scales_non_zero, (1, -1)))

        return CPTensor((weights, normalized_factors))

    @staticmethod
    def cp_flip_sign(cp_tensor, mode=0, func=None):
        """
        Return cp_tensor with factors flipped to have positive signs. The sign of a
        given column is determined by `func`, which is the mean by default. Any negative
        signs are assigned to the mode indicated by `mode`.

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is list of matrices, all with the same number of columns
            i.e.::

                for u in U:
                    u[i].shape == (s_i, R)

            where `R` is fixed while `s_i` can vary with `i`

        mode: int
            mode that should receive negative signs

        func: tensorly function
            a function that should summarize the sign of a column
            it must be able to take an axis argument

        Returns
        -------
        CPTensor = (normalisation_weights, normalised_factors)
        """
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor

        if func is None:
            func = ivy.mean

        for jj in range(0, len(factors)):
            # Skip the target mode
            if jj == mode:
                continue

            # Calculate the sign of the current factor in each component
            column_signs = ivy.sign(func(factors[jj], axis=0))

            # Update both the current and receiving factor
            factors[mode] = factors[mode] * column_signs[ivy.newaxis, :]
            factors[jj] = factors[jj] * column_signs[ivy.newaxis, :]

        # Check the weight signs
        weight_signs = ivy.sign(weights)
        factors[mode] = factors[mode] * weight_signs[ivy.newaxis, :]
        weights = ivy.abs(weights)

        return CPTensor((weights, factors))

    @staticmethod
    def cp_lstsq_grad(cp_tensor, tensor, return_loss=False, mask=None):
        r"""
        Compute (for a third-order tensor)

        .. math::

            \nabla 0.5 ||\\mathcal{X} - [\\mathbf{w}; \\mathbf{A}, \\mathbf{B}, \\mathbf{C}]||^2 # noqa

        where :math:`[\\mathbf{w}; \\mathbf{A}, \\mathbf{B}, \\mathbf{C}]`
        is the CP decomposition with weights
        :math:`\\mathbf{w}` and factor matrices :math:`\\mathbf{A}`, :math:`\\mathbf{B}` and :math:`\\mathbf{C}`. # noqa
        Note that this does not return the gradient
        with respect to the weights even if CP is normalized.

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is a list of factor matrices,
            all with the same number of columns
            i.e. for all matrix U in factor_matrices:
            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

        mask : ndarray
            A mask to be applied to the final tensor. It should be
            broadcastable to the shape of the final tensor, that is
            ``(U[1].shape[0], ... U[-1].shape[0])``.

        return_loss : bool
            Optionally return the scalar loss function along with the gradient.

        Returns
        -------
        cp_gradient : CPTensor = (None, factors)
            factors is a list of factor matrix gradients,
            all with the same number of columns
            i.e. for all matrix U in factor_matrices:
            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

        loss : float
            Scalar quantity of the loss function corresponding to cp_gradient. Only returned
            if return_loss = True.
        """
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        _, factors = cp_tensor

        diff = tensor - ivy.CPTensor.cp_to_tensor(cp_tensor)

        if mask is not None:
            diff = diff * mask

        grad_fac = [
            -ivy.unfolding_dot_khatri_rao(diff, cp_tensor, ii)
            for ii in range(len(factors))
        ]

        if return_loss:
            return CPTensor((None, grad_fac)), 0.5 * ivy.sum(diff**2)

        return CPTensor((None, grad_fac))

    @staticmethod
    def cp_to_tensor(cp_tensor, mask=None):
        """
        Turn the Khatri-product of matrices into a full tensor.

            ``factor_matrices = [|U_1, ... U_n|]`` becomes
            a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is a list of factor matrices,
            all with the same number of columns
            i.e. for all matrix U in factor_matrices:
            U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

        mask : ndarray a mask to be applied to the final tensor. It should be
            broadcastable to the shape of the final tensor, that is
            ``(U[1].shape[0], ... U[-1].shape[0])``.

        Returns
        -------
        ndarray
            full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``

        Notes
        -----
        This version works by first computing the mode-0 unfolding of the tensor
        and then refolding it.

        There are other possible and equivalent alternate implementation, e.g.
        summing over r and updating an outer product of vectors.
        """
        shape, _ = ivy.CPTensor.validate_cp_tensor(cp_tensor)

        if not shape:  # 0-order tensor
            return cp_tensor

        weights, factors = cp_tensor
        if len(shape) == 1:  # just a vector
            return ivy.sum(weights * factors[0], axis=1)

        if weights is None:
            weights = 1

        if mask is None:
            full_tensor = ivy.matmul(
                factors[0] * weights,
                ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=0), (1, 0)),
            )
        else:
            full_tensor = ivy.sum(
                ivy.khatri_rao([factors[0] * weights] + factors[1:], mask=mask), axis=1
            )

        return ivy.fold(full_tensor, 0, shape)

    @staticmethod
    def cp_to_unfolded(cp_tensor, mode):
        """
        Turn the khatri-product of matrices into an unfolded tensor.

            turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`
            unfolding of the tensor

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is a list of matrices, all with the same number of columns
            ie for all u in factor_matrices:
            u[i] has shape (s_u_i, R), where R is fixed
        mode: int
            mode of the desired unfolding

        Returns
        -------
        ndarray
            unfolded tensor of shape (tensor_shape[mode], -1)

        Notes
        -----
        Writing factors = [U_1, ..., U_n], we exploit the fact that
        ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``
        """
        ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor

        if weights is not None:
            return ivy.dot(
                factors[mode] * weights,
                ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=mode), (1, 0)),
            )
        else:
            return ivy.dot(
                factors[mode],
                ivy.permute_dims(ivy.khatri_rao(factors, skip_matrix=mode), (1, 0)),
            )

    @staticmethod
    def cp_to_vec(cp_tensor):
        """
        Turn the khatri-product of matrices into a vector.

            (the tensor ``factors = [|U_1, ... U_n|]``
            is converted into a raveled mode-0 unfolding)

        Parameters
        ----------
        cp_tensor : CPTensor = (weight, factors)
            factors is a list of matrices, all with the same number of columns
            i.e.::

                for u in U:
                    u[i].shape == (s_i, R)

            where `R` is fixed while `s_i` can vary with `i`

        Returns
        -------
        ndarray
            vectorised tensor
        """
        return ivy.tensor_to_vec(ivy.CPTensor.cp_to_tensor(cp_tensor))

    @staticmethod
    def cp_mode_dot(cp_tensor, matrix_or_vector, mode, keep_dim=False, copy=False):
        """
        N-mode product of a CP tensor and a matrix or vector at the specified mode.

        Parameters
        ----------
        cp_tensor : tl.CPTensor or (core, factors)

        matrix_or_vector : ndarray
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode : int

        Returns
        -------
        CPTensor = (core, factors)
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
              if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`
              if matrix_or_vector is a vector

        See Also
        --------
        cp_multi_mode_dot : chaining several mode_dot in one call
        """
        shape, _ = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor
        contract = False

        ndims = len(matrix_or_vector.shape)
        if ndims == 2:  # Tensor times matrix
            # Test for the validity of the operation
            if matrix_or_vector.shape[1] != shape[mode]:
                raise ValueError(
                    f"shapes {shape} and {matrix_or_vector.shape} not aligned in"
                    f" mode-{mode} multiplication: {shape[mode]} (mode {mode}) !="
                    f" {matrix_or_vector.shape[1]} (dim 1 of matrix)"
                )

        elif ndims == 1:  # Tensor times vector
            if matrix_or_vector.shape[0] != shape[mode]:
                raise ValueError(
                    f"shapes {shape} and {matrix_or_vector.shape} not aligned for"
                    f" mode-{mode} multiplication: {shape[mode]} (mode {mode}) !="
                    f" {matrix_or_vector.shape[0]} (vector size)"
                )
            if not keep_dim:
                contract = True  # Contract over that mode
        else:
            raise ValueError("Can only take n_mode_product with a vector or a matrix.")

        if copy:
            factors = [ivy.copy_array(f) for f in factors]
            weights = ivy.copy_array(weights)

        if contract:
            factor = factors.pop(mode)
            factor = ivy.dot(matrix_or_vector, factor)
            mode = max(mode - 1, 0)
            factors[mode] *= factor
        else:
            factors[mode] = ivy.dot(matrix_or_vector, factors[mode])

        if copy:
            return CPTensor((weights, factors))
        else:
            cp_tensor.shape = tuple(f.shape[0] for f in factors)
            return cp_tensor

    @staticmethod
    def cp_norm(cp_tensor):
        """
        Return the l2 norm of a CP tensor.

        Parameters
        ----------
        cp_tensor : tl.CPTensor or (core, factors)

        Returns
        -------
        l2-norm : int

        Notes
        -----
        This is ||cp_to_tensor(factors)||^2

        You can see this using the fact that
        khatria-rao(A, B)^T x khatri-rao(A, B) = A^T x A  * B^T x B
        """
        _ = ivy.CPTensor.validate_cp_tensor(cp_tensor)
        weights, factors = cp_tensor

        norm = ivy.ones((factors[0].shape[1], factors[0].shape[1]), dtype=factors.dtype)
        for f in factors:
            norm = norm * ivy.dot(ivy.permute_dims(f, (1, 0)), ivy.conj(f))

        if weights is not None:
            # norm = T.dot(T.dot(weights, norm), weights)
            norm = norm * (
                ivy.reshape(weights, (-1, 1)) * ivy.reshape(weights, (1, -1))
            )

        return ivy.sqrt(ivy.sum(norm))

    # uncomment when ivy.congruence_coefficient has been implemented
    # which inturn requires linear_sum_assignment to be implemented.
    # @staticmethod
    # def cp_permute_factors(ref_cp_tensor, tensors_to_permute):
    #    """
    #    Compare factors of a reference cp tensor
    #    with factors of other another tensor
    #    (or list of tensor) in order to match
    #    component order. Permutation occurs on the
    #    columns of factors, minimizing the cosine distance
    #    to reference cp tensor with
    #    scipy Linear Sum Assignment method. The permuted
    #    tensor (or list of tensors) and
    #    list of permutation for each
    #    permuted tensors are returned.

    #    Parameters
    #    ----------
    #    ref_cp_tensor : cp tensor
    #        The tensor that serves as a reference for permutation.
    #    tensors_to_permute : cp tensor or list of cp tensors
    #        The tensors to permute so that the order of components
    #        match the reference tensor. Number of components must match.

    #    Returns
    #    -------
    #    permuted_tensors : permuted cp tensor or list of cp tensors
    #    permutation : list
    #        list of permuted indices. Lenght is equal to rank of cp_tensors.
    #    """
    #    if not isinstance(tensors_to_permute, list):
    #        permuted_tensors = [tensors_to_permute.cp_copy()]
    #        tensors_to_permute = [tensors_to_permute]
    #    else:
    #        permuted_tensors = []
    #        for i in range(len(tensors_to_permute)):
    #            permuted_tensors.append(tensors_to_permute[i].cp_copy())
    #            tensors_to_permute[i] = ivy.CPTensor.cp_normalize(tensors_to_permute[i]) # noqa
    #    ref_cp_tensor = ivy.CPTensor.cp_normalize(ref_cp_tensor)
    #    n_tensors = len(tensors_to_permute)
    #    n_factors = len(ref_cp_tensor.factors)
    #    permutation = []
    #    for i in range(n_tensors):
    #        _, col = ivy.congruence_coefficient(
    #            ref_cp_tensor.factors, tensors_to_permute[i].factors
    #        )
    #        col = ivy.array(col, dtype=ivy.int64)
    #        for f in range(n_factors):
    #            permuted_tensors[i].factors[f] = permuted_tensors[i].factors[f][:, col]
    #        permuted_tensors[i].weights = permuted_tensors[i].weights[col]
    #        permutation.append(col)
    #    if len(permuted_tensors) == 1:
    #        permuted_tensors = permuted_tensors[0]
    #    return permuted_tensors, permutation

    @staticmethod
    def unfolding_dot_khatri_rao(tensor, cp_tensor, mode):
        """
        Mode-n unfolding times khatri-rao product of factors.

        Parameters
        ----------
        tensor : tl.tensor
            tensor to unfold
        factors : tl.tensor list
            list of matrices of which to the khatri-rao product
        mode : int
            mode on which to unfold `tensor`

        Returns
        -------
        mttkrp
            dot(unfold(tensor, mode), khatri-rao(factors))
        """
        mttkrp_parts = []
        weights, factors = cp_tensor
        rank = ivy.shape(factors[0])[1]
        for r in range(rank):
            component = ivy.multi_mode_dot(
                tensor, [ivy.conj(f[:, r]) for f in factors], skip=mode
            )
            mttkrp_parts.append(component)

        if weights is None:
            return ivy.stack(mttkrp_parts, axis=1)
        else:
            return ivy.stack(mttkrp_parts, axis=1) * ivy.reshape(weights, (1, -1))
