# global
import abc
from typing import Union, Optional, Literal, Tuple, List, Sequence

# local
import ivy

inf = float("inf")


class _ArrayWithLinearAlgebra(abc.ABC):
    def matmul(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        adjoint_a: bool = False,
        adjoint_b: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matmul. This method simply wraps the
        function, and so the docstring for ivy.matmul also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type. Must have at least one
            dimension.
        x2
            second input array. Should have a numeric data type. Must have at least one
            dimension.
        transpose_a
            if True, ``x1`` is transposed before multiplication.
        transpose_b
            if True, ``x2`` is transposed before multiplication.
        adjoint_a
            If True, takes the conjugate of the matrix then the transpose of the matrix.
            adjoint_a and transpose_a can not be true at the same time.
        adjoint_b
            If True, takes the conjugate of the matrix then the transpose of the matrix.
            adjoint_b and transpose_b can not be true at the same time.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array containing the output of matrix multiplication. The returned array
            must have a data type determined by :ref:`type-promotion`. More details
            can be found in ivy.matmul.

        Examples
        --------
        With :class:`ivy.Array` instance inputs:

        >>> x = ivy.array([1., 4.])
        >>> y = ivy.array([3., 2.])
        >>> z = x.matmul(y)
        >>> print(z)
        ivy.array(11.)
        """
        return ivy.matmul(
            self._data,
            x2,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b,
            out=out,
        )

    def cholesky(
        self: ivy.Array,
        /,
        *,
        upper: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cholesky. This method simply wraps the
        function, and so the docstring for ivy.cholesky also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, M) and whose innermost two dimensions form
            square symmetric positive-definite matrices. Should have a floating-point
            data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the Cholesky factors for each square matrix. If upper is
            False, the returned array must contain lower-triangular matrices; otherwise,
            the returned array must contain upper-triangular matrices. The returned
            array must have a floating-point data type determined by Type Promotion
            Rules and must have the same shape as self.

        Examples
        --------
        >>> x = ivy.array([[4.0, 1.0, 2.0, 0.5, 2.0],
        ...               [1.0, 0.5, 0.0, 0.0, 0.0],
        ...               [2.0, 0.0, 3.0, 0.0, 0.0],
        ...               [0.5, 0.0, 0.0, 0.625, 0.0],
        ...               [2.0, 0.0, 0.0, 0.0, 16.0]])
        >>> y = x.cholesky(upper='false')
        >>> print(y)
        ivy.array([[ 2.  ,  0.5 ,  1.  ,  0.25,  1.  ],
        ...        [ 0.  ,  0.5 , -1.  , -0.25, -1.  ],
        ...        [ 0.  ,  0.  ,  1.  , -0.5 , -2.  ],
        ...        [ 0.  ,  0.  ,  0.  ,  0.5 , -3.  ],
        ...        [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ]])
        """
        return ivy.cholesky(self._data, upper=upper, out=out)

    def cross(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.cross. This method simply wraps the
        function, and so the docstring for ivy.cross also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to
            compute (default: -1) the cross product.vIf set to -1, the function
            computes the cross product for vectors defined by the last axis (dimension).
            Default: ``-1``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise products. The returned array must
            have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`ivy.Array` instance inputs:

        >>> x = ivy.array([1., 0., 0.])
        >>> y = ivy.array([0., 1., 0.])
        >>> z = x.cross(y)
        >>> print(z)
        ivy.array([0., 0., 1.])
        """
        return ivy.cross(self._data, x2, axis=axis, out=out)

    def det(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        Examples
        --------
        >>> x = ivy.array([[2.,4.],[6.,7.]])
        >>> y = x.det()
        >>> print(y)
        ivy.array(-10.)
        """
        return ivy.det(self._data, out=out)

    def diagonal(
        self: ivy.Array,
        /,
        *,
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diagonal. This method simply wraps the
        function, and so the docstring for ivy.diagonal also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices.
        offset
            offset specifying the off-diagonal relative to the main diagonal.
            - ``offset = 0``: the main diagonal.
            - ``offset > 0``: off-diagonal above the main diagonal.
            - ``offset < 0``: off-diagonal below the main diagonal.
            Default: `0`.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from
            which the diagonals should be taken. Defaults to first axis (-2).
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which
            the diagonals should be taken. Defaults to second axis (-1).
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the diagonals and whose shape is determined
            by removing the last two dimensions and appending a dimension equal
            to the size of the resulting diagonals. The returned array must
            have the same data type as ``x``.


        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> x = ivy.array([[1., 2.],
        ...                [3., 4.]])

        >>> d = x.diagonal()
        >>> print(d)
        ivy.array([1., 4.])

        >>> x = ivy.array([[[1., 2.],
        ...                 [3., 4.]],
        ...                [[5., 6.],
        ...                 [7., 8.]]])
        >>> d = x.diagonal()
        >>> print(d)
        ivy.array([[1., 4.],
                   [5., 8.]])

        >>> x = ivy.array([[1., 2.],
        ...                [3., 4.]])

        >>> d = x.diagonal(offset=1)
        >>> print(d)
        ivy.array([2.])

        >>> x = ivy.array([[0, 1, 2],
        ...                [3, 4, 5],
        ...                [6, 7, 8]])
        >>> d = x.diagonal(offset=-1, axis1=0)
        >>> print(d)
        ivy.array([3, 7])
        """
        return ivy.diagonal(
            self._data, offset=offset, axis1=axis1, axis2=axis2, out=out
        )

    def diag(
        self: ivy.Array,
        /,
        *,
        k: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diag. This method simply wraps the
        function, and so the docstring for ivy.diag also applies to this method with
        minimal changes.

        Examples
        --------
        >>> x = ivy.array([[0, 1, 2],
        >>>                [3, 4, 5],
        >>>                [6, 7, 8]])
        >>> x.diag(k=1)
        ivy.array([1, 5])
        """
        return ivy.diag(self._data, k=k, out=out)

    def eig(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> Tuple[ivy.Array]:
        return ivy.eig(self._data, out=out)

    def eigh(
        self: ivy.Array,
        /,
        *,
        UPLO: str = "L",
        out: Optional[ivy.Array] = None,
    ) -> Tuple[ivy.Array]:
        return ivy.eigh(self._data, UPLO=UPLO, out=out)

    def eigvalsh(
        self: ivy.Array,
        /,
        *,
        UPLO: str = "L",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.eigvalsh. This method simply wraps the
        function, and so the docstring for ivy.eigvalsh also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input array having shape (..., M, M) and whose innermost two dimensions form
            square matrices. Must have floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the computed eigenvalues. The returned array must have
            shape (..., M) and have the same data type as x.


        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
        the `docstring <https://data-apis.org/array-api/latest/
        extensions/generated/array_api.linalg.eigvalsh.html>`_
        in the standard.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also
        accepts :class:`ivy.Container` instances in place of any of the arguments.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> x = ivy.array([[[1.0,2.0],[2.0,1.0]]])
        >>> y = ivy.eigvalsh(x)
        >>> print(y)
        ivy.array([[-1.,  3.]])
        """
        return ivy.eigvalsh(self._data, UPLO=UPLO, out=out)

    def inner(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Return the inner product of two vectors ``self`` and ``x2``.

        Parameters
        ----------
        self
            first one-dimensional input array of size N.
            Should have a numeric data type.
            a(N,) array_like
            First input vector. Input is flattened if not already 1-dimensional.
        x2
            second one-dimensional input array of size M.
            Should have a numeric data type.
            b(M,) array_like
            Second input vector. Input is flattened if not already 1-dimensional.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a two-dimensional array containing the inner product and whose
            shape is (N, M).
            The returned array must have a data type determined by Type Promotion Rules.

        Examples
        --------
        Matrices of identical shapes
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> y = ivy.array([[5., 6.], [7., 8.]])
        >>> d = x.inner(y)
        >>> print(d)
        ivy.array([[17., 23.], [39., 53.]])

        Matrices of different shapes
        >>> x = ivy.array([[1., 2.], [3., 4.],[5., 6.]])
        >>> y = ivy.array([[5., 6.], [7., 8.]])
        >>> d = x.inner(y)
        >>> print(d)
        ivy.array([[17., 23.], [39., 53.], [61., 83.]])

        3D matrices
        >>> x = ivy.array([[[1., 2.], [3., 4.]],
                           [[5., 6.], [7., 8.]]])
        >>> y = ivy.array([[[9., 10.], [11., 12.]],
                           [[13., 14.], [15., 16.]]])
        >>> d = x.inner(y)
        >>> print(d)
        ivy.array([[[[ 29.,  35.], [ 41.,  47.]],
                    [[ 67.,  81.], [ 95., 109.]]],
                   [[[105., 127.], [149., 171.]],
                    [[143., 173.], [203., 233.]]]])
        """
        return ivy.inner(self._data, x2, out=out)

    def inv(
        self: ivy.Array, /, *, adjoint: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.inv. This method simply wraps the
        function, and so the docstring for ivy.inv also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape ``(..., M, M)`` and whose innermost two
            dimensions form square matrices. Should have a floating-point data type.

        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the multiplicative inverses. The returned array
            must have a floating-point data type determined by :ref:`type-promotion`
            and must have the same shape as ``x``.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> x = ivy.array([[1.0, 2.0],[3.0, 4.0]])
        >>> y = x.inv()
        >>> print(y)
        ivy.array([[-2., 1.],[1.5, -0.5]])
        """
        return ivy.inv(self._data, adjoint=adjoint, out=out)

    def matrix_norm(
        self: ivy.Array,
        /,
        *,
        ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"]] = "fro",
        axis: Tuple[int, int] = (-2, -1),
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matrix_norm. This method simply wraps
        the function, and so the docstring for ivy.matrix_norm also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices. Should have a floating-point data type.
        ord
            Order of the norm. Default is "fro".
        axis
            specifies the axes that hold 2-D matrices. Default: (-2, -1).
        keepdims
            If this is set to True, the axes which are normed over are left in
            the result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is False.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        Examples
        --------
        >>> x = ivy.array([[1.1, 2.2, 3.3], [1.0, 2.0, 3.0]])
        >>> y = x.matrix_norm(ord=1)
        >>> print(y)
        ivy.array(6.3)

        >>> x = ivy.arange(8, dtype=float).reshape((2, 2, 2))
        >>> y = x.matrix_norm(ord="nuc", keepdims=True)
        >>> print(y)
        ivy.array([[[ 4.24]],
                [[11.4 ]]])
        """
        return ivy.matrix_norm(
            self._data, ord=ord, axis=axis, keepdims=keepdims, out=out
        )

    def matrix_power(
        self: ivy.Array,
        n: int,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.matrix_power(self._data, n, out=out)

    def matrix_rank(
        self: ivy.Array,
        /,
        *,
        atol: Optional[Union[float, Tuple[float]]] = None,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        hermitian: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matrix_rank. This method returns the
        rank (i.e., number of non-zero singular values) of a matrix (or a stack of
        matrices).

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two dimensions
            form ``MxN`` matrices. Should have a floating-point data type.

        atol
            absolute tolerance. When None it’s considered to be zero.

        rtol
            relative tolerance for small singular values. Singular values approximately
            less than or equal to ``rtol * largest_singular_value`` are set to zero.
            If a ``float``, the value is equivalent to a zero-dimensional array having
            a floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``) and must be broadcast against each matrix.
            If an ``array``, must have a floating-point data type and must be
            compatible with ``shape(x)[:-2]`` (see :ref:`broadcasting`).
            If ``None``, the default value is ``max(M, N) * eps``, where ``eps`` must
            be the machine epsilon associated with the floating-point data type
            determined by :ref:`type-promotion` (as applied to ``x``).
            Default: ``None``.

        hermitian
            indicates whether ``x`` is Hermitian. When ``hermitian=True``, ``x`` is
            assumed to be Hermitian, enabling a more efficient method for finding
            eigenvalues, but x is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
            Default: ``False``.

        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and
            must have shape ``(...)``
            (i.e., must have a shape equal to ``shape(x)[:-2]``).

        Examples
        --------
        1. Full Matrix
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> ivy.matrix_rank(x)
        ivy.array(2.)

        2. Rank Deficient Matrix
        >>> x = ivy.array([[1., 0.], [0., 0.]])
        >>> ivy.matrix_rank(x)
        ivy.array(1.)

        3. 1 Dimension - rank 1 unless all 0
        >>> x = ivy.array([[1., 1.])
        >>> ivy.matrix_rank(x)
        ivy.array(1.)

        >>> x = ivy.array([[0., 0.])
        >>> ivy.matrix_rank(x)
        ivy.array(0)
        """
        return ivy.matrix_rank(
            self._data, atol=atol, rtol=rtol, hermitian=hermitian, out=out
        )

    def matrix_transpose(
        self: ivy.Array, /, *, conjugate: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        Transpose a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the transpose for each matrix and having shape
            ``(..., N, M)``. The returned array must have the same data
            type as ``x``.

        Examples
        --------
        With :class:`ivy.Array` instance inputs:

        >>> x = ivy.array([[1., 2.], [0., 3.]])
        >>> y = x.matrix_transpose()
        >>> print(y)
        ivy.array([[1., 0.],
                   [2., 3.]])
        """
        return ivy.matrix_transpose(self._data, conjugate=conjugate, out=out)

    def outer(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Compute the outer product between two arrays.

        Parameters
        ----------
        self : ivy.Array
            The first input array.
        x2 : ivy.Array or ivy.NativeArray
            The second input array.
        out : ivy.Array, optional
            Output array. If provided, it must have the same
            shape as the expected output.

        Returns
        -------
        ivy.Array
            The outer product of the two arrays.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([4, 5])
        >>> z = x.outer(y)
        >>> print(z)
        ivy.array([[ 4,  5],
                   [ 8, 10],
                   [12, 15]])
        """
        return ivy.outer(self._data, x2, out=out)

    def pinv(
        self: ivy.Array,
        /,
        *,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.pinv. This method simply wraps the
        function, and so the docstring for ivy.pinv also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        rtol
            relative tolerance for small singular values. More details in ivy.pinv.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array containing the pseudo-inverses. More details in ivy.pinv.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> y = x.pinv()
        >>> print(y)
        ivy.array([[-1.99999988,  1.        ],
               [ 1.5       , -0.5       ]])

        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> z = ivy.zeros((2,2))
        >>> x.pinv(rtol=0, out=z)
        >>> print(z)
        ivy.array([[-1.99999988,  1.        ],
               [ 1.5       , -0.5       ]])
        """
        return ivy.pinv(self._data, rtol=rtol, out=out)

    def qr(
        self: ivy.Array,
        /,
        *,
        mode: str = "reduced",
        out: Optional[Tuple[ivy.Array, ivy.Array]] = None,
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.qr. This method simply wraps the
        function, and so the docstring for ivy.qr also applies to this method with
        minimal changes.

        Returns the qr decomposition x = QR of a full column rank matrix (or a stack of
        matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an
        upper-triangular matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices of rank N. Should have a floating-point data type.
        mode
            decomposition mode. Should be one of the following modes:
            - 'reduced': compute only the leading K columns of q, such that q and r have
            dimensions (..., M, K) and (..., K, N), respectively, and where
            K = min(M, N).
            - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N),
            respectively.
            Default: 'reduced'.
        out
            optional output tuple of arrays, for writing the result to. The arrays must
            have shapes that the inputs broadcast to.

        Returns
        -------
        ret
            a namedtuple (Q, R) whose
            - first element must have the field name Q and must be an array whose shape
            depends on the value of mode and contain matrices with orthonormal columns.
            If mode is 'complete', the array must have shape (..., M, M). If mode is
            'reduced', the array must have shape (..., M, K), where K = min(M, N). The
            first x.ndim-2 dimensions must have the same size as those of the input
            array x.
            - second element must have the field name R and must be an array whose shape
            depends on the value of mode and contain upper-triangular matrices. If mode
            is 'complete', the array must have shape (..., M, N). If mode is 'reduced',
            the array must have shape (..., K, N), where K = min(M, N). The first
            x.ndim-2 dimensions must have the same size as those of the input x.
        """
        return ivy.qr(self._data, mode=mode, out=out)

    def slogdet(
        self: ivy.Array,
    ) -> Tuple[ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.slogdet. This method simply wraps the
        function, and so the docstring for ivy.slogdet also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, M) and whose innermost two dimensions
            form square matrices. Should have a floating-point data type.

        Returns
        -------
        ret
            This function returns NamedTuple with two values -
                sign:
                An array containing a number representing the sign of the determinant
                for each square matrix.

                logabsdet:
                An array containing natural log of the absolute determinant of each
                square matrix.

        Examples
        --------
        >>> x = ivy.array([[1.0, 2.0],
        ...                [3.0, 4.0]])
        >>> y = x.slogdet()
        >>> print(y)
        slogdet(sign=ivy.array(-1.), logabsdet=ivy.array(0.69314718))

        >>> x = ivy.array([[1.2, 2.0, 3.1],
        ...                [6.0, 5.2, 4.0],
        ...                [9.0, 8.0, 7.0]])
        >>> y = x.slogdet()
        >>> print(y)
        slogdet(sign=ivy.array(-1.), logabsdet=ivy.array(1.098611))
        """
        return ivy.slogdet(self._data)

    def solve(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        adjoint: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.solve(self._data, x2, adjoint=adjoint, out=out)

    def svd(
        self: ivy.Array,
        /,
        *,
        compute_uv: bool = True,
        full_matrices: bool = True,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
        """
        ivy.Array instance method variant of ivy.svf. This method simply wraps the
        function, and so the docstring for ivy.svd also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form matrices on which to perform singular value decomposition.
            Should have a floating-point data type.
        full_matrices
            If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has shape
            ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``,
            compute on the leading ``K`` singular vectors, such that ``U`` has
            shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where
            ``K = min(M, N)``. Default: ``True``.
        compute_uv
            If ``True`` then left and right singular vectors will be computed and
            returned in ``U`` and ``Vh``, respectively. Otherwise, only the
            singular values will be computed, which can be significantly faster.
        .. note::
            with backend set as torch, svd with still compute left and right singular
            vectors irrespective of the value of compute_uv, however Ivy will still only
            return the singular values.

        Returns
        -------
        .. note::
            once complex numbers are supported, each square matrix must be Hermitian.

        ret
            a namedtuple ``(U, S, Vh)``. More details in ivy.svd.

            Each returned array must have the same floating-point data type as ``x``.

        Examples
        --------
        With :class:`ivy.Array` input:

        >>> x = ivy.random_normal(shape = (9, 6))
        >>> U, S, Vh = x.svd()
        >>> print(U.shape, S.shape, Vh.shape)
        (9, 9) (6,) (6, 6)

        With reconstruction from SVD, result is numerically close to x

        >>> reconstructed_x = ivy.matmul(U[:,:6] * S, Vh)
        >>> print((reconstructed_x - x > 1e-3).sum())
        ivy.array(0)

        >>> U, S, Vh = x.svd(full_matrices = False)
        >>> print(U.shape, S.shape, Vh.shape)
        (9, 6) (6,) (6, 6)
        """
        return ivy.svd(self._data, compute_uv=compute_uv, full_matrices=full_matrices)

    def svdvals(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.svdvals(self._data, out=out)

    def tensordot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.tensordot(self._data, x2, axes=axes, out=out)

    def tensorsolve(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axes: Optional[Union[int, Tuple[List[int], List[int]]]] = None,
    ) -> Tuple[ivy.Array]:
        return ivy.tensorsolve(self._data, x2, axes=axes)

    def trace(
        self: ivy.Array,
        /,
        *,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.trace. This method Returns the sum
        along the specified diagonals of a matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        offset
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative. Defaults to 0.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the traces and whose shape is determined by removing
            the last two dimensions and storing the traces in the last array dimension.
            For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``,
            then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array must have the same data type as ``x``.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> y = x.trace()
        >>> print(y)
        ivy.array(5.)

        >>> x = ivy.array([[1., 2., 4.], [6., 5., 3.]])
        >>> y = ivy.Array.trace(x)
        >>> print(y)
        ivy.array(6.)
        """
        return ivy.trace(self._data, offset=offset, axis1=axis1, axis2=axis2, out=out)

    def vecdot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vecdot(self._data, x2, axis=axis, out=out)

    def vector_norm(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        ord: Union[int, float, Literal[inf, -inf]] = 2,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.vector_norm. This method computes the
        vector norm of a vector (or batch of vectors).

        Parameters
        ----------
        self
            Input array. Should have a floating-point data type.
        axis
            If an integer, ``axis`` specifies the axis (dimension) along which to
            compute vector norms. If an n-tuple, ``axis`` specifies the axes
            (dimensions) along which to compute batched vector norms. If ``None``,
            the vector norm must be computed over all array values (i.e., equivalent
            to computing the vector norm of a flattened array). Negative indices are
            also supported. Default: ``None``.
        keepdims
            If ``True``, the axes (dimensions) specified by ``axis`` must be included
            in the result as singleton dimensions, and, accordingly, the result must be
            compatible with the input array (see :ref:`broadcasting`). Otherwise, if
            ``False``, the axes (dimensions) specified by ``axis`` must not be included
            in the result.
            Default: ``False``.
        ord
            order of the norm. The following mathematical norms are supported:

            +------------------+----------------------------+
            | ord              | description                |
            +==================+============================+
            | 1                | L1-norm (Manhattan)        |
            +------------------+----------------------------+
            | 2                | L2-norm (Euclidean)        |
            +------------------+----------------------------+
            | inf              | infinity norm              |
            +------------------+----------------------------+
            | (int,float >= 1) | p-norm                     |
            +------------------+----------------------------+

            The following non-mathematical "norms" are also supported:

            +------------------+--------------------------------+
            | ord              | description                    |
            +==================+================================+
            | 0                | sum(a != 0)                    |
            +------------------+--------------------------------+
            | -inf             | min(abs(a))                    |
            +------------------+--------------------------------+
            | (int,float < 1)  | sum(abs(a)**ord)**(1./ord)     |
            +------------------+--------------------------------+

            Default: ``2``.
        dtype
            data type that may be used to perform the computation more precisely.
            The input array ``self`` gets cast to ``dtype`` before the function's
            computations.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the vector norms. If ``axis`` is ``None``, the returned
            array must be a zero-dimensional array containing a vector norm. If ``axis``
            is a scalar value (``int`` or ``float``), the returned array must have a
            rank which is one less than the rank of ``self``. If ``axis`` is a
            ``n``-tuple, the returned array must have a rank which is ``n`` less than
            the rank of ``self``. The returned array must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> y = x.vector_norm()
        >>> print(y)
        ivy.array([3.7416575])
        """
        return ivy.vector_norm(
            self._data, axis=axis, keepdims=keepdims, ord=ord, dtype=dtype, out=out
        )

    def vector_to_skew_symmetric_matrix(
        self: ivy.Array, /, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.vector_to_skew_symmetric_matrix(self._data, out=out)

    def vander(
        self: ivy.Array,
        /,
        *,
        N: Optional[int] = None,
        increasing: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.vander. This method Returns the
        Vandermonde matrix of the input array.

        Parameters
        ----------
        self
            1-D input array.
        N
            Number of columns in the output. If N is not specified,
            a square array is returned (N = len(x))
        increasing
            Order of the powers of the columns. If True, the powers increase
            from left to right, if False (the default) they are reversed.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            an array containing the Vandermonde matrix.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, 5])
        >>> ivy.vander(x)
        ivy.array(
        [[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]]
            )

        >>> x = ivy.array([1, 2, 3, 5])
        >>> ivy.vander(x, N=3)
        ivy.array(
        [[ 1,  1,  1],
            [ 4,  2,  1],
            [ 9,  3,  1],
            [25,  5,  1]]
            )

        >>> x = ivy.array([1, 2, 3, 5])
        >>> ivy.vander(x, N=3, increasing=True)
        ivy.array(
        [[ 1,  1,  1],
            [ 1,  2,  4],
            [ 1,  3,  9],
            [ 1,  5, 25]]
            )
        """
        return ivy.vander(self._data, N=N, increasing=increasing, out=out)
