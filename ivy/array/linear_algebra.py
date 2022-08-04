# global
import abc
from typing import Union, Optional, Literal, NamedTuple, Tuple, List

# local
import ivy

inf = float("inf")


class ArrayWithLinearAlgebra(abc.ABC):
    def matmul(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        """
        Examples
        ------------------------

        With :code:`ivy.Array` instance inputs:

        >>> x = ivy.array([1., 4.])
        >>> y = ivy.array([3., 2.])
        >>> z = x.matmul(y)
        >>> print(z)
        ivy.array(11.)
        """        
        return ivy.matmul(self._data, x2, out=out)

    def cholesky(
        self: ivy.Array,
        upper: bool = False,
        *,
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
            Default: False.
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
        >>> x = ivy.array([[4.0, 1.0, 2.0, 0.5, 2.0], \
                       [1.0, 0.5, 0.0, 0.0, 0.0], \
                       [2.0, 0.0, 3.0, 0.0, 0.0], \
                       [0.5, 0.0, 0.0, 0.625, 0.0], \
                       [2.0, 0.0, 0.0, 0.0, 16.0]])
        >>> y = x.cholesky('false')
        >>> print(y)
        ivy.array([[ 2.  ,  0.5 ,  1.  ,  0.25,  1.  ],
                   [ 0.  ,  0.5 , -1.  , -0.25, -1.  ],
                   [ 0.  ,  0.  ,  1.  , -0.5 , -2.  ],
                   [ 0.  ,  0.  ,  0.  ,  0.5 , -3.  ],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  1.  ]])
        """
        return ivy.cholesky(self._data, upper, out=out)

    def cross(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        *,
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
            Default: -1.
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
        With :code:`ivy.Array` inputs:
        
        >>> x = ivy.array([1., 0., 0.])
        >>> y = ivy.array([0., 1., 0.])
        >>> z = x.cross(y)
        >>> print(z)
        ivy.array([0., 0., 1.])
        """
        return ivy.cross(self._data, x2, axis, out=out)

    def det(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.det(self._data, out=out)

    def diagonal(
        self: ivy.Array,
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.diagonal(self._data, offset, axis1, axis2, out=out)

    def eigh(
        self: ivy.Array,
    ) -> NamedTuple:
        return ivy.eigh(self._data)

    def eigvalsh(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.eigvalsh(self._data, out=out)

    def inv(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.inv(self._data, out=out)

    def matrix_norm(
        self: ivy.Array,
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.matrix_norm(self._data, ord, keepdims, out=out)

    def matrix_rank(
        self: ivy.Array,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.matrix_rank(self._data, rtol, out=out)
        """
        ivy.Array instance method variant of ivy.matrix_rank. This method returns 
        the rank (i.e., number of non-zero singular values) of a matrix (or a stack of
        matrices).

        Parameters
        ----------
        self
            input array having shape ``(..., M, N)`` and whose innermost two dimensions 
            form ``MxN`` matrices. Should have a floating-point data type.
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

    def matrix_transpose(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.matrix_transpose(self._data, out=out)

    def outer(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.outer(self._data, x2, out=out)

    def pinv(
        self: ivy.Array,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.pinv(self._data, rtol, out=out)

    def qr(
        self: ivy.Array,
        mode: str = "reduced",
    ) -> NamedTuple:
        return ivy.qr(self._data, mode)

    def solve(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.solve(self._data, x2, out=out)

    def svd(
        self: ivy.Array,
        full_matrices: bool = True,
    ) -> Union[ivy.Array, Tuple[ivy.Array, ...]]:
        return ivy.svd(self._data, full_matrices)

    def svdvals(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.svdvals(self._data, out=out)

    def tensordot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.tensordot(self._data, x2, axes, out=out)

    def trace(
        self: ivy.Array,
        offset: int = 0,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.trace(self._data, offset, out=out)

    def vecdot(
        self: ivy.Array,
        x2: Union[ivy.Array, ivy.NativeArray],
        axis: int = -1,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vecdot(self._data, x2, axis, out=out)

    def vector_norm(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        ord: Union[int, float, Literal[inf, -inf]] = 2,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.vector_norm(self._data, axis, keepdims, ord, out=out)

    def vector_to_skew_symmetric_matrix(
        self: ivy.Array, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.vector_to_skew_symmetric_matrix(self._data, out=out)
