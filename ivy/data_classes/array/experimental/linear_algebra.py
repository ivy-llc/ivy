# global
import abc
from typing import Optional, Union, Tuple, List, Sequence

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

    def cov(
        self: ivy.Array,
        x2: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        /,
        *,
        rowVar: bool = True,
        bias: bool = False,
        ddof: Optional[int] = None,
        fweights: Optional[ivy.Array] = None,
        aweights: Optional[ivy.Array] = None,
        dtype: Optional[type] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cov. This method simply wraps the
        function, and so the docstring for ivy.cov also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            a 1D or 2D input array, with a numeric data type.
        x2
            optional second 1D or 2D input array, with a numeric data type.
            Must have the same shape as ``self``.
        rowVar
            optional variable where each row of input is interpreted as a variable
            (default = True). If set to False, each column is instead interpreted as a
            variable.
        bias
            optional variable for normalizing input (default = False) by (N - 1) where
            N is the number of given observations. If set to True, then normalization
            is instead by N. Can be overridden by keyword ``ddof``.
        ddof
            optional variable to override ``bias`` (default = None). ddof=1 will return
            the unbiased estimate, even with fweights and aweights given. ddof=0 will
            return the simple average.
        fweights
            optional 1D array of integer frequency weights; the number of times each
            observation vector should be repeated.
        aweights
            optional 1D array of observation vector weights. These relative weights are
            typically large for observations considered "important" and smaller for
            observations considered less "important". If ddof=0 is specified, the array
            of weights can be used to assign probabilities to observation vectors.
        dtype
            optional variable to set data-type of the result. By default, data-type
            will have at least ``float64`` precision.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the covariance matrix of an input matrix, or the
            covariance matrix of two variables. The returned array must have a
            floating-point data type determined by Type Promotion Rules and must be
            a square matrix of shape (N, N), where N is the number of variables in the
            input(s).

        Examples
        --------
        >>> x = ivy.array([[1,2,3],
        ...                [4,5,6]])
        >>> y = x.cov()
        >>> print(y)
        ivy.array([[ 1.,  1.  ],
        ...        [ 1.,  1.  ],
        >>> x = ivy.array([1,2,3])
        >>> y = ivy.array([4,5,6])
        >>> z = x.cov(y)
        >>> print(z)
        ivy.array([[ 1.,  1.  ],
        ...        [ 1.,  1.  ])
        """
        return ivy.cov(
            self._data,
            x2,
            rowVar=rowVar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
            dtype=dtype,
        )
