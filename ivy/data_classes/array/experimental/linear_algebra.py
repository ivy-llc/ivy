# global
import abc
from typing import Optional, Union, Tuple, List, Sequence, Literal

# local
import ivy


class _ArrayWithLinearAlgebraExperimental(abc.ABC):
    def eigh_tridiagonal(
        self: Union[ivy.Array, ivy.NativeArray],
        beta: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        eigvals_only: bool = True,
        select: str = "a",
        select_range: Optional[
            Union[Tuple[int, int], List[int], ivy.Array, ivy.NativeArray]
        ] = None,
        tol: Optional[float] = None,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ivy.Array]]:
        """
        ivy.Array instance method variant of ivy.eigh_tridiagonal. This method simply
        wraps the function, and so the docstring for ivy.eigh_tridiagonal also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            An array of real or complex arrays each of shape (n),
            the diagonal elements of the matrix.
        beta
            An array or of real or complex arrays each of shape (n-1),
            containing the elements of the first super-diagonal of the matrix.
        eigvals_only
            If False, both eigenvalues and corresponding eigenvectors are computed.
            If True, only eigenvalues are computed. Default is True.
        select
            Optional string with values in {'a', 'v', 'i'}
            (default is 'a') that determines which eigenvalues
            to calculate: 'a': all eigenvalues. 'v': eigenvalues
            in the interval (min, max] given by select_range.
            'i': eigenvalues with indices min <= i <= max.
        select_range
            Size 2 tuple or list or array specifying the range of
            eigenvalues to compute together with select. If select
            is 'a', select_range is ignored.
        tol
            Optional scalar. Ignored when backend is not Tensorflow. The
            absolute tolerance to which each eigenvalue is required. An
            eigenvalue (or cluster) is considered to have converged if
            it lies in an interval of this width. If tol is None (default),
            the value eps*|T|_2 is used where eps is the machine precision,
            and |T|_2 is the 2-norm of the matrix T.

        Returns
        -------
        eig_vals
            The eigenvalues of the matrix in non-decreasing order.
        eig_vectors
            If eigvals_only is False the eigenvectors are returned in the second
            output argument.

        Examples
        --------
        >>> alpha = ivy.array([0., 1., 2.])
        >>> beta = ivy.array([0., 1.])
        >>> y = alpha.eigh_tridiagonal(beta)
        >>> print(y)
        ivy.array([0., 0.38196, 2.61803])
        """
        return ivy.eigh_tridiagonal(
            self._data,
            beta,
            eigvals_only=eigvals_only,
            select=select,
            select_range=select_range,
            tol=tol,
        )

    def diagflat(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        offset: int = 0,
        padding_value: float = 0,
        align: str = "RIGHT_LEFT",
        num_rows: int = -1,
        num_cols: int = -1,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diagflat. This method simply wraps the
        function, and so the docstring for ivy.diagflat also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = ivy.array([1,2])
        >>> x.diagflat(k=1)
        ivy.array([[0, 1, 0],
                   [0, 0, 2],
                   [0, 0, 0]])
        """
        return ivy.diagflat(
            self._data,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            out=out,
        )

    def kron(
        self: ivy.Array,
        b: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.kron. This method simply wraps the
        function, and so the docstring for ivy.kron also applies to this method with
        minimal changes.

        Examples
        --------
        >>> a = ivy.array([1,2])
        >>> b = ivy.array([3,4])
        >>> a.diagflat(b)
        ivy.array([3, 4, 6, 8])
        """
        return ivy.kron(self._data, b, out=out)

    def matrix_exp(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.kron. This method simply wraps the
        function, and so the docstring for ivy.matrix_exp also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([[[1., 0.],
                            [0., 1.]],
                            [[2., 0.],
                            [0., 2.]]])
        >>> ivy.matrix_exp(x)
        ivy.array([[[2.7183, 1.0000],
                    [1.0000, 2.7183]],
                    [[7.3891, 1.0000],
                    [1.0000, 7.3891]]])
        """
        return ivy.matrix_exp(self._data, out=out)

    def eig(
        self: ivy.Array,
        /,
    ) -> Tuple[ivy.Array, ...]:
        """
        ivy.Array instance method variant of ivy.eig. This method simply wraps the
        function, and so the docstring for ivy.eig also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.eig()
        (
        ivy.array([-0.37228132+0.j,  5.37228132+0.j]),
        ivy.array([[-0.82456484+0.j, -0.41597356+0.j],
                   [ 0.56576746+0.j, -0.90937671+0.j]])
        )
        """
        return ivy.eig(self._data)

    def eigvals(
        self: ivy.Array,
        /,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.eigvals. This method simply wraps the
        function, and so the docstring for ivy.eigvals also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.eigvals()
        ivy.array([-0.37228132+0.j,  5.37228132+0.j])
        """
        return ivy.eigvals(self._data)

    def adjoint(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.adjoint. This method simply wraps the
        function, and so the docstring for ivy.adjoint also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = np.array([[1.-1.j, 2.+2.j],
                          [3.+3.j, 4.-4.j]])
        >>> x = ivy.array(x)
        >>> x.adjoint()
        ivy.array([[1.+1.j, 3.-3.j],
                   [2.-2.j, 4.+4.j]])
        """
        return ivy.adjoint(
            self._data,
            out=out,
        )

    def multi_dot(
        self: ivy.Array,
        x: Sequence[Union[ivy.Array, ivy.NativeArray]],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.multi_dot. This method simply wraps the
        function, and so the docstring for ivy.multi_dot also applies to this method
        with minimal changes.

        Examples
        --------
        >>> A = ivy.arange(2 * 3).reshape((2, 3))
        >>> B = ivy.arange(3 * 2).reshape((3, 2))
        >>> C = ivy.arange(2 * 2).reshape((2, 2))
        >>> A.multi_dot((B, C))
        ivy.array([[ 26,  49],
                   [ 80, 148]])
        """
        return ivy.multi_dot((self._data, *x), out=out)

    def cond(
        self: ivy.Array, /, *, p: Optional[Union[int, float, str]] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cond. This method simply wraps the
        function, and so the docstring for ivy.cond also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.cond()
        ivy.array(14.933034373659268)

        >>> x = ivy.array([[1,2], [3,4]])
        >>> x.cond(p=ivy.inf)
        ivy.array(21.0)
        """
        return ivy.cond(self._data, p=p)

    def mode_dot(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        matrix_or_vector: Union[ivy.Array, ivy.NativeArray],
        mode: int,
        transpose: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.mode_dot. This method simply wraps the
        function, and so the docstring for ivy.mode_dot also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            tensor of shape ``(i_1, ..., i_k, ..., i_N)``
        matrix_or_vector
            1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
            matrix or vectors to which to n-mode multiply the tensor
        mode
            int in the range(1, N)
        transpose
            If True, the matrix is transposed.
            For complex tensors, the conjugate transpose is used.
        out
            optional output array, for writing the result to.
            It must have a shape that the result can broadcast to.

        Returns
        -------
        ivy.Array
            `mode`-mode product of `tensor` by `matrix_or_vector`
            * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)`
            if matrix_or_vector is a matrix
            * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)`
            if matrix_or_vector is a vector
        """
        return ivy.mode_dot(self._data, matrix_or_vector, mode, transpose, out=out)

    def multi_mode_dot(
        self: Union[ivy.Array, ivy.NativeArray],
        mat_or_vec_list: Sequence[Union[ivy.Array, ivy.NativeArray]],
        /,
        modes: Optional[Sequence[int]] = None,
        skip: Optional[Sequence[int]] = None,
        transpose: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        r"""
        ivy.Array instance method variant of ivy.multi_mode_dot. This method simply
        wraps the function, and so the docstring for ivy.multi_mode_dot also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            the input tensor

        mat_or_vec_list
            sequence of matrices or vectors of length ``tensor.ndim``

        skip
            None or int, optional, default is None
            If not None, index of a matrix to skip.

        modes
            None or int list, optional, default is None

        transpose
            If True, the matrices or vectors in in the list are transposed.
            For complex tensors, the conjugate transpose is used.
        out
            optional output array, for writing the result to. It must have a shape that the
            result can broadcast to.

        Returns
        -------
        ivy.Array
            tensor times each matrix or vector in the list at mode `mode`

        Notes
        -----
        If no modes are specified, just assumes there is one matrix or vector per mode and returns:
        :math:`\\text{x  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }` # noqa
        """
        return ivy.multi_mode_dot(
            self._data, mat_or_vec_list, modes, skip, transpose, out=out
        )

    def svd_flip(
        self: Union[ivy.Array, ivy.NativeArray],
        V: Union[ivy.Array, ivy.NativeArray],
        /,
        u_based_decision: Optional[bool] = True,
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.svd_flip. This method simply wraps the
        function, and so the docstring for ivy.svd_flip also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            left singular matrix output of SVD
        V
            right singular matrix output of SVD
        u_based_decision
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.

        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        return ivy.svd_flip(self._data, V, u_based_decision)

    def make_svd_non_negative(
        self: Union[ivy.Array, ivy.NativeArray],
        U: Union[ivy.Array, ivy.NativeArray],
        S: Union[ivy.Array, ivy.NativeArray],
        V: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        nntype: Optional[Literal["nndsvd", "nndsvda"]] = "nndsvd",
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.make_svd_non_negative. This method
        simply wraps the function, and so the docstring for ivy.make_svd_non_negative
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            tensor being decomposed.
        U
            left singular matrix from SVD.
        S
            diagonal matrix from SVD.
        V
            right singular matrix from SVD.
        nntype
            whether to fill small values with 0.0 (nndsvd),
            or the tensor mean (nndsvda, default).

        [1]: Boutsidis & Gallopoulos. Pattern Recognition, 41(4): 1350-1362, 2008.
        """
        return ivy.make_svd_non_negative(self._data, U, S, V, nntype=nntype)

    def truncated_svd(
        self: Union[ivy.Array, ivy.NativeArray],
        /,
        compute_uv: bool = True,
        n_eigenvecs: Optional[int] = None,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ivy.Array, ivy.Array]]:
        """
        ivy.Array instance method variant of ivy.make_svd_non_negative. This method
        simply wraps the function, and so the docstring for ivy.make_svd_non_negative
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            2D-array
        compute_uv
            If ``True`` then left and right singular vectors will
            be computed and returnedv in ``U`` and ``Vh``
            respectively. Otherwise, only the singular values will
            be computed, which can be significantly faster.
        n_eigenvecs
            if specified, number of eigen[vectors-values] to return
            else full matrices will be returned

        Returns
        -------
        ret
            a namedtuple ``(U, S, Vh)``
            Each returned array must have the same floating-point data type as ``x``.
        """
        return ivy.truncated_svd(self._data, compute_uv, n_eigenvecs)

    def initialize_tucker(
        self: Union[ivy.Array, ivy.NativeArray],
        rank: Sequence[int],
        modes: Sequence[int],
        /,
        *,
        init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
        seed: Optional[int] = None,
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        non_negative: Optional[bool] = False,
        mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
    ) -> Tuple[ivy.Array, Sequence[ivy.Array]]:
        """
        ivy.Array instance method variant of ivy.initialize_tucker. This method simply
        wraps the function, and so the docstring for ivy.initialize_tucker also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input tensor
        rank
            number of components
        modes
            modes to consider in the input tensor
        seed
            Used to create a random seed distribution
            when init == 'random'
        init
            initialization scheme for tucker decomposition.
        svd
            function to use to compute the SVD
        non_negative
            if True, non-negative factors are returned
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None

        Returns
        -------
        core
            initialized core tensor
        factors
            list of factors
        """
        return ivy.initialize_tucker(
            self._data,
            rank,
            modes,
            seed=seed,
            init=init,
            svd=svd,
            non_negative=non_negative,
            mask=mask,
            svd_mask_repeats=svd_mask_repeats,
        )

    def partial_tucker(
        self: Union[ivy.Array, ivy.NativeArray],
        rank: Optional[Sequence[int]] = None,
        modes: Optional[Sequence[int]] = None,
        /,
        *,
        n_iter_max: Optional[int] = 100,
        init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        seed: Optional[int] = None,
        mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
        tol: Optional[float] = 10e-5,
        verbose: Optional[bool] = False,
        return_errors: Optional[bool] = False,
    ) -> Tuple[ivy.Array, Sequence[ivy.Array]]:
        """
        ivy.Array instance method variant of ivy.partial_tucker. This method simply
        wraps the function, and so the docstring for ivy.partial_tucker also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            the  input tensor
        rank
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
            if int, the same rank is used for all modes
            if None, original tensors size will be preserved.
        modes
            list of the modes on which to perform the decomposition
        n_iter_max
            maximum number of iteration
        init
            {'svd', 'random'}, or TuckerTensor optional
            if a TuckerTensor is provided, this is used for initialization
        svd
            str, default is 'truncated_svd'
            function to use to compute the SVD,
        seed
            Used to create a random seed distribution
            when init == 'random'
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None
        tol
            tolerance: the algorithm stops when the variation in
            the reconstruction error is less than the tolerance.
        verbose
            if True, different in reconstruction errors are returned at each
            iteration.
        return_erros
            if True, list of reconstruction errors are returned.

        Returns
        -------
        core : ndarray
                core tensor of the Tucker decomposition
        factors : ndarray list
                list of factors of the Tucker decomposition.
                with ``core.shape[i] == (tensor.shape[i], ranks[i]) for i in modes``
        """
        return ivy.partial_tucker(
            self._data,
            rank,
            modes,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            seed=seed,
            mask=mask,
            svd_mask_repeats=svd_mask_repeats,
            tol=tol,
            verbose=verbose,
            return_errors=return_errors,
        )

    def tucker(
        self: Union[ivy.Array, ivy.NativeArray],
        rank: Optional[Sequence[int]] = None,
        /,
        *,
        fixed_factors: Optional[Sequence[int]] = None,
        n_iter_max: Optional[int] = 100,
        init: Optional[Union[Literal["svd", "random"], ivy.TuckerTensor]] = "svd",
        svd: Optional[Literal["truncated_svd"]] = "truncated_svd",
        seed: Optional[int] = None,
        mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        svd_mask_repeats: Optional[int] = 5,
        tol: Optional[float] = 10e-5,
        verbose: Optional[bool] = False,
        return_errors: Optional[bool] = False,
    ):
        """
        ivy.Array instance method variant of ivy.tucker. This method simply wraps the
        function, and so the docstring for ivy.tucker also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input tensor
        rank
            size of the core tensor, ``(len(ranks) == tensor.ndim)``
            if int, the same rank is used for all modes
        fixed_factors
            if not None, list of modes for which to keep the factors fixed.
            Only valid if a Tucker tensor is provided as init.
        n_iter_max
            maximum number of iteration
        init
            {'svd', 'random'}, or TuckerTensor optional
            if a TuckerTensor is provided, this is used for initialization
        svd
            str, default is 'truncated_svd'
            function to use to compute the SVD,
        seed
            Used to create a random seed distribution
            when init == 'random'
        mask
            array of booleans with the same shape as ``tensor`` should be 0 where
            the values are missing and 1 everywhere else. Note:  if tensor is
            sparse, then mask should also be sparse with a fill value of 1 (or
            True).
        svd_mask_repeats
            number of iterations for imputing the values in the SVD matrix when
            mask is not None
        tol
            tolerance: the algorithm stops when the variation in
            the reconstruction error is less than the tolerance
        verbose
            if True, different in reconstruction errors are returned at each
            iteration.

        return_errors
            Indicates whether the algorithm should return all reconstruction errors
            and computation time of each iteration or not
            Default: False


        Returns
        -------
            ivy.TuckerTensor or ivy.TuckerTensor and
            list of reconstruction errors if return_erros is True.

        References
        ----------
        .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
        SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
        """
        return ivy.tucker(
            self._data,
            rank,
            fixed_factors=fixed_factors,
            n_iter_max=n_iter_max,
            init=init,
            return_errors=return_errors,
            seed=seed,
            mask=mask,
            svd=svd,
            svd_mask_repeats=svd_mask_repeats,
            tol=tol,
            verbose=verbose,
        )
