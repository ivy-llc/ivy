# global
import abc
from typing import Union, Optional, Literal, Tuple, List, Sequence

# local
import ivy

inf = float("inf")


class ArrayWithLinearAlgebra(abc.ABC):
    def matmul(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Examples
        ------------------------

        With :class:`ivy.Array` instance inputs:

        >>> x = ivy.array([1., 4.])
        >>> y = ivy.array([3., 2.])
        >>> z = x.matmul(y)
        >>> print(z)
        ivy.array(11.)
        """
        return ivy.matmul(
            self._data, x2, transpose_a=transpose_a, transpose_b=transpose_b, out=out
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
        function, and so the docstring for ivy.cholesky also applies to this method
        with minimal changes.

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
        function, and so the docstring for ivy.cross also applies to this method
        with minimal changes.

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
        ivy.Array instance method variant of ivy.diag.
        This method simply wraps the function, and so the docstring for
        ivy.diag also applies to this method with minimal changes.

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
        UPLO: Optional[str] = "L",
        out: Optional[ivy.Array] = None,
    ) -> Tuple[ivy.Array]:
        return ivy.eigh(self._data, UPLO=UPLO, out=out)

    def eigvalsh(
        self: ivy.Array,
        /,
        *,
        UPLO: Optional[str] = "L",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.eigvalsh.
        This method simply wraps the function, and so the docstring for
        ivy.eigvalsh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array having shape (..., M, M) and whose innermost two dimensions form
            square matrices. Must have floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the computed eigenvalues. The returned array must have shape
            (..., M) and have the same data type as x.


        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
        `docstring <https://data-apis.org/array-api/latest/extensions/generated/signatures.linalg.eigvalsh.html>`_ # noqa
        in the standard.

        Both the description and the type hints above assumes an array input for simplicity,
        but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
        instances in place of any of the arguments.

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
        return ivy.inner(self._data, x2, out=out)

    def inv(
        self: ivy.Array, /, *, adjoint: bool = False, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.inv.
        This method simply wraps the function, and so the docstring for
        ivy.inv also applies to this method with minimal changes.

        Examples
        --------
        With :class:`ivy.Array` inputs:

        >>> x = ivy.array([[1.0, 2.0],[3.0, 4.0]])
        >>> y = ivy.inv(x)
        >>> print(y)
        ivy.array([[-2., 1.],[1.5, -0.5]])
        """
        return ivy.inv(self._data, adjoint=adjoint, out=out)

    def matrix_norm(
        self: ivy.Array,
        /,
        *,
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        axis: Optional[Tuple[int, int]] = (-2, -1),
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matrix_norm.
        This method simply wraps the function, and so the docstring for
        ivy.matrix_norm also applies to this method with minimal changes.

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
        >>> y = x.matrix_norm(ord="nuc", axis=(2, 1), keepdims=True)
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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.matrix_rank. This method returns
        the rank (i.e., number of non-zero singular values) of a matrix (or a stack of
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
        return ivy.matrix_rank(self._data, atol=atol, rtol=rtol, out=out)

    def matrix_transpose(
        self: ivy.Array, /, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        Transposes a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        x
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
        >>> y = ivy.matrix_transpose(x)
        >>> print(y)
        ivy.array([[1., 0.],
                   [2., 3.]])
        """
        return ivy.matrix_transpose(self._data, out=out)

    def outer(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.outer(self._data, x2, out=out)

    def pinv(
        self: ivy.Array,
        /,
        *,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> y = x.pinv()
        >>> print(y)
        ivy.array([[-2., 1.],
                   [1.5, -0.5]])

        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> z = ivy.zeros((2,2))
        >>> x.pinv(rtol=0, out=z)
        >>> print(z)
        ivy.array([[0.0426, 0.0964],
                   [0.0605, 0.1368]])
        """
        return ivy.pinv(self._data, rtol=rtol, out=out)

    def qr(
        self: ivy.Array,
        /,
        *,
        mode: str = "reduced",
    ) -> Tuple[ivy.Array, ivy.Array]:
        return ivy.qr(self._data, mode=mode)

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
        slogdet(sign=ivy.array(-1.), logabsdet=ivy.array(0.6931472))

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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.solve(self._data, x2, out=out)

    def svd(
        self: ivy.Array,
        /,
        *,
        compute_uv: bool = True,
        full_matrices: bool = True,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
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
        axes: Union[int, Tuple[List[int], List[int]]] = None,
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
        ivy.Array instance method variant of ivy.trace.
        This method Returns the sum along the specified diagonals of a matrix (or a
        stack of matrices).

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
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vector_norm(
            self._data, axis=axis, keepdims=keepdims, ord=ord, out=out
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
        increasing: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.vander.
        This method Returns the Vandermonde matrix of the input array.

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
