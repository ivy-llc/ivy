# global

from typing import Union, Optional, Tuple, Literal, List, Dict, Sequence

# local
from ivy.data_classes.container.base import ContainerBase
import ivy

inf = float("inf")


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor,PyMethodParameters
class _ContainerWithLinearAlgebra(ContainerBase):
    @staticmethod
    def _static_matmul(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        transpose_a: Union[bool, ivy.Container] = False,
        transpose_b: Union[bool, ivy.Container] = False,
        adjoint_a: Union[bool, ivy.Container] = False,
        adjoint_b: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.matmul. This method
        simply wraps the function, and so the docstring for ivy.matul also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array
        x2
            second input array
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            the matrix multiplication result of x1 and x2

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = ivy.array([[2., 1.], [1., 1.]]))
        >>> y = ivy.Container.static_matmul(x, x)
        >>> print(y)
        {
            a: ivy.array([[10., -6.],
                          [-6., 10.]]),
            b: ivy.array([[5., 3.],
                          [3., 2.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matmul",
            x1,
            x2,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matmul(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        transpose_a: Union[bool, ivy.Container] = False,
        transpose_b: Union[bool, ivy.Container] = False,
        adjoint_a: Union[bool, ivy.Container] = False,
        adjoint_b: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.matmul. This method
        simply wraps the function, and so the docstring for ivy.matmul also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array
        x2
            second input array
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            the matrix multiplication result of self and x2

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = ivy.array([[2., 1.], [1., 1.]]))
        >>> y = x.matmul(x)
        >>> print(y)
        {
            a: ivy.array([[10., -6.],
                          [-6., 10.]]),
            b: ivy.array([[5., 3.],
                          [3., 2.]])
        }
        """
        return self._static_matmul(
            self,
            x2,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cholesky(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        upper: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.cholesky. This method
        simply wraps the function, and so the docstring for ivy.cholesky also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container having shape (..., M, M) and whose innermost two
            dimensions form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the Cholesky factors for each square matrix. If upper
            is False, the returned container must contain lower-triangular matrices;
            otherwise, the returned container must contain upper-triangular matrices.
            The returned container must have a floating-point data type determined by
            Type Promotion Rules and must have the same shape as self.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[3., -1.], [-1., 3.]]),
        ...                      b=ivy.array([[2., 1.], [1., 1.]]))
        >>> y = ivy.Container.static_cholesky(x, upper='false')
        >>> print(y)
        {
            a: ivy.array([[1.73, -0.577],
                            [0., 1.63]]),
            b: ivy.array([[1.41, 0.707],
                            [0., 0.707]])
         }
        With multiple :class:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.array([[3., -1], [-1., 3.]]),
        ...                      b=ivy.array([[2., 1.], [1., 1.]]))
        >>> upper = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, upper=False)
        >>> print(y)
        {
            a: ivy.array([[3., 3.],
                         [-1., -1.]]),
            b: ivy.array([[1., 1.],
                          [1., 2.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cholesky",
            x,
            upper=upper,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cholesky(
        self: ivy.Container,
        /,
        *,
        upper: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.cholesky. This method
        simply wraps the function, and so the docstring for ivy.cholesky also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container having shape (..., M, M) and whose innermost two dimensions
            form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the Cholesky factors for each square matrix. If upper
            is False, the returned container must contain lower-triangular matrices;
            otherwise, the returned container must contain upper-triangular matrices.
            The returned container must have a floating-point data type determined by
            Type Promotion Rules and must have the same shape as self.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[3., -1],[-1., 3.]]),
        ...                      b=ivy.array([[2., 1.],[1., 1.]]))
        >>> y = x.cholesky(upper='false')
        >>> print(y)
        {
            a: ivy.array([[1.73, -0.577],
                            [0., 1.63]]),
            b: ivy.array([[1.41, 0.707],
                            [0., 0.707]])
        }
        """
        return self._static_cholesky(
            self,
            upper=upper,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cross(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.cross. This method simply
        wraps the function, and so the docstring for ivy.cross also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to
            compute the cross product.vIf set to -1, the function computes the
            cross product for vectors defined by the last axis (dimension).
            Default: ``-1``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise products. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.array([9., 0., 3.])
        >>> y = ivy.Container(a=ivy.array([1., 1., 0.]), b=ivy.array([1., 0., 1.]))
        >>> z = ivy.Container.static_cross(x, y)
        >>> print(z)
        {
            a: ivy.array([-3., 3., 9.]),
            b: ivy.array([0., -6., 0.])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = x = ivy.Container(a=ivy.array([5., 0., 0.]), b=ivy.array([0., 0., 2.]))
        >>> y = ivy.Container(a=ivy.array([0., 7., 0.]), b=ivy.array([3., 0., 0.]))
        >>> z = ivy.Container.static_cross(x, y)
        >>> print(z)
        {
            a: ivy.array([0., 0., 35.]),
            b: ivy.array([0., 6., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cross",
            x1,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cross(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.cross. This method
        simply wraps the function, and so the docstring for ivy.cross also
        applies to this method with minimal changes.

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
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise products. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([5., 0., 0.]), b=ivy.array([0., 0., 2.]))
        >>> y = ivy.Container(a=ivy.array([0., 7., 0.]), b=ivy.array([3., 0., 0.]))
        >>> z = x.cross(y)
        >>> print(z)
        {
            a: ivy.array([0., 0., 35.]),
            b: ivy.array([0., 6., 0.])
        }
        """
        return self._static_cross(
            self,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_det(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "det",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def det(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = ivy.array([[2., 1.], [1., 1.]]))
        >>> y = x.det()
        >>> print(y)
        {a:ivy.array(8.),b:ivy.array(1.)}
        """
        return self._static_det(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_diagonal(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        offset: Union[int, ivy.Container] = 0,
        axis1: Union[int, ivy.Container] = -2,
        axis2: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.diagonal. This method
        simply wraps the function, and so the docstring for ivy.diagonal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input Container with leave arrays having shape
             ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices.
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
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken. Defaults to second axis (-1).
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the diagonals. More details can be found in
            the docstring for ivy.diagonal.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[1., 2.], [3., 4.]],
        ...                   b=ivy.array([[5., 6.], [7., 8.]])))
        >>> d = ivy.Container.static_diagonal(x)
        >>> print(d)
        {
            a:ivy.array([1., 4.]),
            b:ivy.array([5., 8.])
        }

        >>> a = ivy.array([[0, 1, 2],
        ...                [3, 4, 5],
        ...                [6, 7, 8]])
        >>> b = ivy.array([[-1., -2., -3.],
        ...                 [-3., 4., 5.],
        ...                 [5., 6., 7.]])],
        >>> x = ivy.Container(a=a, b=b)
        >>> d = ivy.Container.static_diagonal(offset=-1, axis1=0)
        >>> print(d)
        {
            a:ivy.array([3., 7.]),
            b:ivy.array([-3., 6.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "diagonal",
            x,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diagonal(
        self: ivy.Container,
        /,
        *,
        offset: Union[int, ivy.Container] = 0,
        axis1: Union[int, ivy.Container] = -2,
        axis2: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.diagonal. This method
        simply wraps the function, and so the docstring for ivy.diagonal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input Container with leave arrays having shape
             ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices.
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
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken. Defaults to second axis (-1).
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the diagonals. More details can be found in
            the docstring for ivy.diagonal.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[1., 2.], [3., 4.]]),
        ...                   b=ivy.array([[5., 6.], [7., 8.]]))
        >>> d = x.diagonal()
        >>> print(d)
        {
            a:ivy.array([1., 4.]),
            b:ivy.array([5., 8.])
        }

        >>> a = ivy.array([[0, 1, 2],
        ...                [3, 4, 5],
        ...                [6, 7, 8]])
        >>> b = ivy.array([[-1., -2., -3.],
        ...                 [-3., 4., 5.],
        ...                 [5., 6., 7.]]),
        >>> x = ivy.Container(a=a, b=b)
        >>> d = x.diagonal(offset=-1)
        >>> print(d)
        {
            a: ivy.array([3, 7]),
            b: ivy.array([[-3., 6.]])
        }
        """
        return self._static_diagonal(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_diag(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        k: Union[int, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "diag",
            x,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diag(
        self: ivy.Container,
        /,
        *,
        k: Union[int, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.diag. This method
        simply wraps the function, and so the docstring for ivy.diag also
        applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=[[0, 1, 2],
        >>>                      [3, 4, 5],
        >>>                      [6, 7, 8]])
        >>> ivy.diag(x, k=1)
        {
            a: ivy.array([1, 5])
        }
        """
        return self._static_diag(
            self,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_eigh(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        UPLO: Union[str, ivy.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "eigh",
            x,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def eigh(
        self: ivy.Container,
        /,
        *,
        UPLO: Union[str, ivy.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.eigh. This method
        simply wraps the function, and so the docstring for ivy.eigh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self : ivy.Container
            Ivy container having shape `(..., M, M)` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        UPLO : str, optional
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix should be
            used for the eigenvalue decomposition. Default is 'L'.
        key_chains : Union[List[str], Dict[str, str]], optional
            The key-chains to apply or not apply the method to. Default is `None`.
        to_apply : bool, optional
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is `True`.
        prune_unapplied : bool, optional
            Whether to prune key_chains for which the function was not applied.
            Default is `False`.
        map_sequences : bool, optional
            Whether to also map method to sequences (lists, tuples).
            Default is `False`.
        out : ivy.Container, optional
            Optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ivy.Container
            A container containing the computed eigenvalues.
            The returned array must have shape `(..., M)` and have the same
            data type as `self`.

        Examples
        --------
        With `ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[[1.,2.],[2.,1.]]]),
        ...                   b=ivy.array([[[2.,4.],[4.,2.]]]))
        >>> y = x.eigh()
        >>> print(y)
        {
            a: ivy.array([[-1., 3.]]),
            b: ivy.array([[-2., 6.]])
        }
        """
        return self._static_eigh(
            self,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_eigvalsh(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        UPLO: Union[str, ivy.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.eigvalsh. This method
        simply wraps the function, and so the docstring for ivy.eigvalsh also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Ivy container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the computed eigenvalues.
            The returned array must have shape
            (..., M) and have the same data type as x.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]]),
        ...                   b=ivy.array([[[1.,1.,2.],[1.,2.,1.],[2.,1.,1.]]]),
        ...                   c=ivy.array([[[2.,2.,2.],[2.,3.,3.],[2.,3.,3.]]]))
        >>> e = ivy.Container.static_eigvalsh(x)
        >>> print(e)
        {
            a: ivy.array([[-0.51572949, 0.17091519, 11.3448143]]),
            b: ivy.array([[-1., 1., 4.]]),
            c: ivy.array([[-8.88178420e-16, 5.35898387e-01, 7.46410179e+00]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "eigvalsh",
            x,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def eigvalsh(
        self: ivy.Container,
        /,
        *,
        UPLO: Union[str, ivy.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.eigvalsh. This method
        simply wraps the function, and so the docstring for ivy.eigvalsh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Ivy container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the computed eigenvalues.
            The returned array must have shape
            (..., M) and have the same data type as x.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([[[1.,2.],[2.,1.]]]),
        ...                   b=ivy.array([[[2.,4.],[4.,2.]]]))
        >>> y = ivy.eigvalsh(x)
        >>> print(y)
        {
            a: ivy.array([[-1., 3.]]),
            b: ivy.array([[-2., 6.]])
        }
        """
        return self._static_eigvalsh(
            self,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inner(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.inner. This method simply
        wraps the function, and so the docstring for ivy.inner also applies to
        this method with minimal changes.

        Return the inner product of two vectors ``x1`` and ``x2``.

        Parameters
        ----------
        x1
            first one-dimensional input array of size N.
            Should have a numeric data type.
            a(N,) array_like
            First input vector. Input is flattened if not already 1-dimensional.
        x2
            second one-dimensional input array of size M.
            Should have a numeric data type.
            b(M,) array_like
            Second input vector. Input is flattened if not already 1-dimensional.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
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
        >>> x1 = ivy.Container(a=ivy.array([[1, 2], [3, 4]]))
        >>> x2 = ivy.Container(a=ivy.array([5, 6]))
        >>> y = ivy.Container.static_inner(x1, x2)
        >>> print(y)
        {
            a: ivy.array([17, 39])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inner",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def inner(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.inner. This method
        simply wraps the function, and so the docstring for ivy.inner also
        applies to this method with minimal changes.

        Return the inner product of two vectors ``self`` and ``x2``.

        Parameters
        ----------
        self
            input container of size N. Should have a numeric data type.
            a(N,) array_like
            First input vector. Input is flattened if not already 1-dimensional.
        x2
            one-dimensional input array of size M. Should have a numeric data type.
            b(M,) array_like
            Second input vector. Input is flattened if not already 1-dimensional.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a new container representing the inner product and whose
            shape is (N, M).
            The returned array must have a data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([[1, 2], [3, 4]]))
        >>> x2 = ivy.Container(a=ivy.array([5, 6]))
        >>> y = ivy.Container.inner(x1, x2)
        >>> print(y)
        {
            a: ivy.array([17, 39])
        }
        """
        return self._static_inner(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inv(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        adjoint: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.inv. This method simply
        wraps the function, and so the docstring for ivy.inv also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Ivy container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container containing the multiplicative inverses.
            The returned array must have a floating-point data type
            determined by :ref:`type-promotion` and must have the
            same shape as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[0., 1.], [4., 4.]]),
        ...                      b=ivy.array([[4., 4.], [2., 1.]]))
        >>> y = ivy.Container.static_inv(x)
        >>> print(y)
        {
            a: ivy.array([[-1, 0.25], [1., 0.]]),
            b: ivy.array([-0.25, 1.], [0.5, -1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inv",
            x,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def inv(
        self: ivy.Container,
        /,
        *,
        adjoint: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.inv. This method simply
        wraps the function, and so the docstring for ivy.inv also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Ivy container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container containing the multiplicative inverses.
            The returned array must have a floating-point data type
            determined by :ref:`type-promotion` and must have the
            same shape as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[0., 1.], [4., 4.]]),
        ...                      b=ivy.array([[4., 4.], [2., 1.]]))
        >>> y = x.inv()
        >>> print(y)
        {
            a: ivy.array([[-1, 0.25], [1., 0.]]),
            b: ivy.array([-0.25, 1.], [0.5, -1.])
        }
        """
        return self._static_inv(
            self,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_pinv(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container special method variant of ivy.pinv. This method simply
        wraps the function, and so the docstring for ivy.pinv also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form``MxN`` matrices. Should have a floating-point
            data type.
        rtol
            relative tolerance for small singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the pseudo-inverses. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and
            must have shape ``(..., N, M)`` (i.e., must have the same shape as
            ``x``, except the innermost two dimensions must be transposed).

        Examples
        --------
        >>> x = ivy.Container(a= ivy.array([[1., 2.], [3., 4.]]))
        >>> y = ivy.Container.static_pinv(x)
        >>> print(y)
        {
            a: ivy.array([[-2., 1.],
                          [1.5, -0.5]])
        }

        >>> x = ivy.Container(a=ivy.array([[1., 2.], [3., 4.]]))
        >>> out = ivy.Container(a=ivy.zeros((2, 2)))
        >>> ivy.Container.static_pinv(x, rtol=1e-1, out=out)
        >>> print(out)
        {
            a: ivy.array([[0.0426, 0.0964],
                          [0.0605, 0.1368]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "pinv",
            x,
            rtol=rtol,
            out=out,
        )

    def pinv(
        self: ivy.Container,
        /,
        *,
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.pinv. This method
        simply wraps the function, and so the docstring for ivy.pinv also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array having shape ``(..., M, N)`` and whose innermost
            two dimensions form``MxN`` matrices. Should have a floating-point
            data type.
        rtol
            relative tolerance for small singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the pseudo-inverses. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and
            must have shape ``(..., N, M)`` (i.e., must have the same shape as
            ``x``, except the innermost two dimensions must be transposed).


        Examples
        --------
        >>> x = ivy.Container(a= ivy.array([[1., 2.], [3., 4.]]))
        >>> y = x.pinv()
        >>> print(y)
        {
            a: ivy.array([[-1.99999988, 1.],
                          [1.5, -0.5]])
        }

        >>> x = ivy.Container(a = ivy.array([[1., 2.], [3., 4.]]))
        >>> out = ivy.Container(a = ivy.zeros(x["a"].shape))
        >>> x.pinv(out=out)
        >>> print(out)
        {
            a: ivy.array([[-1.99999988, 1.],
                          [1.5, -0.5]])
        }
        """
        return self._static_pinv(
            self,
            rtol=rtol,
            out=out,
        )

    @staticmethod
    def _static_matrix_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"], ivy.Container] = "fro",
        axis: Tuple[int, int, ivy.Container] = (-2, -1),
        keepdims: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.matrix_norm. This method
        simply wraps the function, and so the docstring for ivy.matrix_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array having shape (..., M, N) and whose innermost two deimensions
            form MxN matrices. Should have a floating-point data type.
        ord
            Order of the norm. Default is "fro".
        axis
            specifies the axes that hold 2-D matrices. Default: (-2, -1).
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is ``False``.
        dtype
            If specified, the input tensor is cast to dtype before performing
            the operation, and the returned tensor's type will be dtype.
            Default: None
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.1, 2.2], [1., 2.]]), \
                              b=ivy.array([[1., 2.], [3., 4.]]))
        >>> y = ivy.Container.static_matrix_norm(x, ord=1)
        >>> print(y)
        {
            a: ivy.array(4.2),
            b: ivy.array(6.)
        }

        >>> x = ivy.Container(a=ivy.arange(12, dtype=float).reshape((3, 2, 2)), \
                              b=ivy.arange(8, dtype=float).reshape((2, 2, 2)))
        >>> ord = ivy.Container(a=1, b=float('inf'))
        >>> axis = ivy.Container(a=(1, 2), b=(2, 1))
        >>> k = ivy.Container(a=False, b=True)
        >>> y = ivy.Container.static_matrix_norm(x, ord=ord, axis=axis, keepdims=k)
        >>> print(y)
        {
            a: ivy.array([4.24, 11.4, 19.2]),
            b: ivy.array([[[3.7]],
                          [[11.2]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_norm",
            x,
            ord=ord,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_norm(
        self: ivy.Container,
        /,
        *,
        ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"], ivy.Container] = "fro",
        axis: Tuple[int, int, ivy.Container] = (-2, -1),
        keepdims: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.matrix_norm. This
        method simply wraps the function, and so the docstring for
        ivy.matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices. Should have a floating-point data type.
        ord
            Order of the norm. Default is "fro".
        axis
            specifies the axes that hold 2-D matrices. Default: (-2, -1).
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is ``False``.
        dtype
            If specified, the input tensor is cast to dtype before performing
            the operation, and the returned tensor's type will be dtype.
            Default: None
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.1, 2.2], [1., 2.]]), \
                              b=ivy.array([[1., 2.], [3., 4.]]))
        >>> y = x.matrix_norm(ord=1)
        >>> print(y)
        {
            a: ivy.array(4.2),
            b: ivy.array(6.)
        }

        >>> x = ivy.Container(a=ivy.arange(12, dtype=float).reshape((3, 2, 2)), \
                              b=ivy.arange(8, dtype=float).reshape((2, 2, 2)))
        >>> ord = ivy.Container(a="nuc", b=ivy.inf)
        >>> axis = ivy.Container(a=(1, 2), b=(2, 1))
        >>> k = ivy.Container(a=True, b=False)
        >>> y = x.matrix_norm(ord=ord, axis=axis, keepdims=k)
        >>> print(y)
        {
            a: ivy.array([[[4.24]],
                         [[11.4]],
                         [[19.2]]]),
            b: ivy.array([4., 12.])
        }
        """
        return self._static_matrix_norm(
            self,
            ord=ord,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_power(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        n: Union[int, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "matrix_power",
            x,
            n,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_power(
        self: ivy.Container,
        n: Union[int, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_matrix_power(
            self,
            n,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_rank(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        atol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        hermitian: Optional[Union[bool, ivy.Container]] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.matrix_rank. This method
        returns the rank (i.e., number of non-zero singular values) of a matrix
        (or a stack of matrices).

        Parameters
        ----------
        x
            input array or container having shape ``(..., M, N)`` and whose innermost
            two dimensions form ``MxN`` matrices. Should have a floating-point data
            type.

        atol
            absolute tolerance. When None it’s considered to be zero.

        rtol
            relative tolerance for small singular values. Singular values
            approximately less than or equal to ``rtol * largest_singular_value`` are
            set to zero. If a ``float``, the value is equivalent to a zero-dimensional
            array having a floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``) and must be broadcast against each matrix. If an
            ``array``, must have a floating-point data type and must be compatible with
            ``shape(x)[:-2]`` (see:ref:`broadcasting`). If ``None``, the default value
            is ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated
            with the floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``).
            Default: ``None``.

        hermitian
            indicates whether ``x`` is Hermitian. When ``hermitian=True``, ``x`` is
            assumed to be Hermitian, enabling a more efficient method for finding
            eigenvalues, but x is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and must have
            shape ``(...)`` (i.e., must have a shape equal to ``shape(x)[:-2]``).

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[1., 0.], [0., 1.]]),
        ...                   b=ivy.array([[1., 0.], [0., 0.]]))
        >>> y = ivy.Container.static_matrix_rank(x)
        >>> print(y)
        {
            a: ivy.array(2.),
            b: ivy.array(1.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_rank",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            atol=atol,
            rtol=rtol,
            hermitian=hermitian,
            out=out,
        )

    def matrix_rank(
        self: ivy.Container,
        /,
        *,
        atol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        hermitian: Optional[Union[bool, ivy.Container]] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.matrix_rank. This
        method returns the rank (i.e., number of non-zero singular values) of a
        matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.

        atol
            absolute tolerance. When None it’s considered to be zero.

        rtol
            relative tolerance for small singular values. Singular values approximately
            less than or equal to ``rtol * largest_singular_value`` are set to zero. If
            a ``float``, the value is equivalent to a zero-dimensional array having a
            floating-point data type determined by :ref:`type-promotion` (as applied to
            ``x``) and must be broadcast against each matrix. If an ``array``, must have
            a floating-point data type and must be compatible with ``shape(x)[:-2]``
            (see :ref:`broadcasting`). If ``None``, the default value is
            ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated
            with the floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``). Default: ``None``.

        hermitian
            indicates whether ``x`` is Hermitian. When ``hermitian=True``, ``x`` is
            assumed to be Hermitian, enabling a more efficient method for finding
            eigenvalues, but x is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and must have
            shape ``(...)`` (i.e., must have a shape equal to ``shape(x)[:-2]``).

        Examples
        --------
        With :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[1., 0.], [0., 1.]]),
        ...                   b=ivy.array([[1., 0.], [0., 0.]]))
        >>> y = x.matrix_rank()
        >>> print(y)
        {
            a: ivy.array(2),
            b: ivy.array(1)
        }
        """
        return self._static_matrix_rank(
            self,
            atol=atol,
            rtol=rtol,
            hermitian=hermitian,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_transpose(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        conjugate: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Transpose a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        x
            input Container which will have arrays with shape ``(..., M, N)``
            and whose innermost two dimensions form ``MxN`` matrices.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the transposes for each matrix and having shape
            ``(..., N, M)``. The returned array must have the same data
            type as ``x``.


        Examples
        --------
        With :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([[1., 1.], [0., 3.]]), \
                        b=ivy.array([[0., 4.], [3., 1.]]))
        >>> y = ivy.Container.static_matrix_transpose(x)
        >>> print(y)
        {
            a: ivy.array([[1., 0.],
                          [1., 3.]]),
            b: ivy.array([[0., 3.],
                          [4., 1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_transpose",
            x,
            conjugate=conjugate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_transpose(
        self: ivy.Container,
        /,
        *,
        conjugate: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Transpose a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        self
            input Container which will have arrays with shape ``(..., M, N)``
            and whose innermost two dimensions form ``MxN`` matrices.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the transposes for each matrix and having shape
            ``(..., N, M)``. The returned array must have the same data
            type as ``x``.

        Examples
        --------
        With :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([[1., 1.], [0., 3.]]), \
                      b=ivy.array([[0., 4.], [3., 1.]]))
        >>> y = x.matrix_transpose()
        >>> print(y)
        {
            a: ivy.array([[1., 0.],
                          [1., 3.]]),
            b: ivy.array([[0., 3.],
                          [4., 1.]])
        }
        """
        return self._static_matrix_transpose(
            self,
            conjugate=conjugate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_outer(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.outer. This method simply
        wraps the function, and so the docstring for ivy.outer also applies to
        this method with minimal changes.

        Computes the outer product of two arrays, x1 and x2,
        by computing the tensor product along the last dimension of both arrays.

        Parameters
        ----------
        x1
            first input array having shape (..., N1)
        x2
            second input array having shape (..., N2)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            The container must have shape (..., N1, N2). The first x1.ndim-1
            dimensions must have the same size as those of the input array x1
            and the first x2.ndim-1 dimensions must have the same
            size as those of the input array x2.

        Returns
        -------
        ret
            an ivy container whose shape is (..., N1, N2).
            The first x1.ndim-1 dimensions have the same size as those
            of the input array x1 and the first x2.ndim-1
            dimensions have the same size as those of the input array x2.

        Example
        -------
        >>> x1 =ivy.Container( a=ivy.array([[1, 2, 3], [4, 5, 6]]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]))
        >>> y = ivy.Container.static_outer(x1, x2)
        >>> print(y)
        ivy.array([[[ 1.,  2.,  3.],
                    [ 2.,  4.,  6.],
                    [ 3.,  6.,  9.]],
                   [[ 4.,  8., 12.],
                    [ 5., 10., 15.],
                    [ 6., 12., 18.]]])
        """
        return ContainerBase.cont_multi_map_in_function(
            "outer",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def outer(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Return the outer product of two arrays or containers.

        The instance method implementation of the static method static_outer of the
        ivy.Container class. It calculates the outer product of two input arrays or
        containers along the last dimension and returns the resulting container. The
        input arrays should be either ivy.Container, ivy.Array, or ivy.NativeArray. The
        output container shape is the concatenation of the shapes of the input
        containers along the last dimension.

        Parameters
        ----------
        self : ivy.Container
            Input container of shape (...,B) where the last dimension
            represents B elements.
        x2 : Union[ivy.Container, ivy.Array, ivy.NativeArray]
            Second input array or container of shape (..., N)
            where the last dimension represents N elements.
        key_chains : Optional[Union[List[str], Dict[str, str]]]
            The key-chains to apply or not apply the method to. Default is None.
        to_apply : bool
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped.Default is True.
        prune_unapplied : bool
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences : bool
            Whether to also map the method to sequences (lists, tuples).
            Default is False.
        out : Optional[ivy.Container]
            Optional output container to write the result to.
            If not provided, a new container will be created.

        Returns
        -------
        ivy.Container
            A new container of shape (..., M, N) representing
            the outer product of the input arrays or containers
            along the last dimension.

        Examples
        --------
        >>> x = ivy.array([[1., 2.],[3., 4.]])
        >>> y = ivy.array([[5., 6.],[7., 8.]])
        >>> d = ivy.outer(x,y)
        >>> print(d)
        ivy.array([[ 5.,  6.,  7.,  8.],
                    [10., 12., 14., 16.],
                    [15., 18., 21., 24.],
                    [20., 24., 28., 32.]])
        """
        return self._static_outer(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_qr(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        mode: Union[str, ivy.Container] = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Tuple[ivy.Container, ivy.Container]] = None,
    ) -> Tuple[ivy.Container, ivy.Container]:
        """ivy.Container static method variant of ivy.qr. This method simply
        wraps the function, and so the docstring for ivy.qr also applies to
        this method with minimal changes.

        Returns the qr decomposition x = QR of a full column rank matrix (or a stack of
        matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an
        upper-triangular matrix (or a stack of matrices).

        Parameters
        ----------
        x
            input container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices of rank N. Should have a floating-point data type.
        mode
            decomposition mode. Should be one of the following modes:
            - 'reduced': compute only the leading K columns of q, such that q and r have
            dimensions (..., M, K) and (..., K, N), respectively, and where
            K = min(M, N).
            - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N),
            respectively.
            Default: 'reduced'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output tuple of containers, for writing the result to. The arrays
            must have shapes that the inputs broadcast to.

        Returns
        -------
        ret
            a namedtuple (Q, R) whose
            - first element must have the field name Q and must be an container whose
            shape depends on the value of mode and contain matrices with orthonormal
            columns. If mode is 'complete', the container must have shape (..., M, M).
            If mode is 'reduced', the container must have shape (..., M, K), where
            K = min(M, N). The first x.ndim-2 dimensions must have the same size as
            those of the input container x.
            - second element must have the field name R and must be an container whose
            shape depends on the value of mode and contain upper-triangular matrices. If
            mode is 'complete', the container must have shape (..., M, N). If mode is
            'reduced', the container must have shape (..., K, N), where K = min(M, N).
            The first x.ndim-2 dimensions must have the same size as those of the input
            x.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.native_array([[1., 2.], [3., 4.]]),
        ...                   b = ivy.array([[2., 3.], [4. ,5.]]))
        >>> q,r = ivy.Container.static_qr(x, mode='complete')
        >>> print(q)
        {
            a: ivy.array([[-0.31622777, -0.9486833],
                        [-0.9486833, 0.31622777]]),
            b: ivy.array([[-0.4472136, -0.89442719],
                        [-0.89442719, 0.4472136]])
        }
        >>> print(r)
        {
            a: ivy.array([[-3.16227766, -4.42718872],
                        [0., -0.63245553]]),
            b: ivy.array([[-4.47213595, -5.81377674],
                        [0., -0.4472136]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "qr",
            x,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def qr(
        self: ivy.Container,
        /,
        *,
        mode: Union[str, ivy.Container] = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Tuple[ivy.Container, ivy.Container]] = None,
    ) -> Tuple[ivy.Container, ivy.Container]:
        """ivy.Container instance method variant of ivy.qr. This method simply
        wraps the function, and so the docstring for ivy.qr also applies to
        this method with minimal changes.

        Returns the qr decomposition x = QR of a full column rank matrix (or a stack of
        matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an
        upper-triangular matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices of rank N. Should have a floating-point data type.
        mode
            decomposition mode. Should be one of the following modes:
            - 'reduced': compute only the leading K columns of q, such that q and r have
            dimensions (..., M, K) and (..., K, N), respectively, and where
            K = min(M, N).
            - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N),
            respectively.
            Default: 'reduced'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output tuple of containers, for writing the result to. The arrays
            must have shapes that the inputs broadcast to.

        Returns
        -------
        ret
            a namedtuple (Q, R) whose
            - first element must have the field name Q and must be an container whose
            shape depends on the value of mode and contain matrices with orthonormal
            columns. If mode is 'complete', the container must have shape (..., M, M).
            If mode is 'reduced', the container must have shape (..., M, K), where
            K = min(M, N). The first x.ndim-2 dimensions must have the same size as
            those of the input container x.
            - second element must have the field name R and must be an container whose
            shape depends on the value of mode and contain upper-triangular matrices. If
            mode is 'complete', the container must have shape (..., M, N). If mode is
            'reduced', the container must have shape (..., K, N), where K = min(M, N).
            The first x.ndim-2 dimensions must have the same size as those of the input
            x.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.native_array([[1., 2.], [3., 4.]]),
        ...                   b = ivy.array([[2., 3.], [4. ,5.]]))
        >>> q,r = x.qr(mode='complete')
        >>> print(q)
        {
            a: ivy.array([[-0.31622777, -0.9486833],
                        [-0.9486833, 0.31622777]]),
            b: ivy.array([[-0.4472136, -0.89442719],
                        [-0.89442719, 0.4472136]])
        }
        >>> print(r)
        {
            a: ivy.array([[-3.16227766, -4.42718872],
                        [0., -0.63245553]]),
            b: ivy.array([[-4.47213595, -5.81377674],
                        [0., -0.4472136]])
        }
        """
        return self._static_qr(
            self,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_slogdet(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.slogdet. This method
        simply wraps the function, and so the docstring for ivy.slogdet also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container having shape (..., M, M) and whose innermost two
            dimensions form square matrices. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            This function returns a container containing NamedTuples.
            Each NamedTuple of output will have -
                sign:
                An array containing a number representing the sign of the determinant
                for each square matrix.

                logabsdet:
                An array containing natural log of the absolute determinant of each
                square matrix.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.0, 2.0],
        ...                                [3.0, 4.0]]),
        ...                   b=ivy.array([[1.0, 2.0],
        ...                                [2.0, 1.0]]))
        >>> y = ivy.Container.static_slogdet(x)
        >>> print(y)
        {
            a: [
                sign = ivy.array(-1.),
                logabsdet = ivy.array(0.6931472)
            ],
            b: [
                sign = ivy.array(-1.),
                logabsdet = ivy.array(1.0986123)
            ]
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "slogdet",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def slogdet(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.slogdet. This method
        simply wraps the function, and so the docstring for ivy.slogdet also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container having shape (..., M, M) and whose innermost two dimensions
            form square matrices. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            This function returns container containing NamedTuples.
            Each NamedTuple of output will have -
                sign:
                An array of a number representing the sign of the determinant of each
                square.

                logabsdet:
                An array of the natural log of the absolute value of the determinant of
                each square.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[1.0, 2.0],
        ...                                [3.0, 4.0]]),
        ...                   b=ivy.array([[1.0, 2.0],
        ...                                [2.0, 1.0]]))
        >>> y = x.slogdet()
        >>> print(y)
        [{
            a: ivy.array(-1.),
            b: ivy.array(-1.)
        }, {
            a: ivy.array(0.69314718),
            b: ivy.array(1.09861231)
        }]
        """
        return self._static_slogdet(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_solve(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        adjoint: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "solve",
            x1,
            x2,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def solve(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        adjoint: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_solve(
            self,
            x2,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_svd(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        compute_uv: Union[bool, ivy.Container] = True,
        full_matrices: Union[bool, ivy.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> Union[ivy.Container, Tuple[ivy.Container, ...]]:
        """ivy.Container static method variant of ivy.svd. This method simply
        wraps the function, and so the docstring for ivy.svd also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container with array leaves having shape ``(..., M, N)`` and whose
            innermost two dimensions form matrices on which to perform singular value
            decomposition. Should have a floating-point data type.
        full_matrices
            If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has
            shape ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``,
            compute on             the leading ``K`` singular vectors, such that ``U``
            has shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where
            ``K = min(M, N)``. Default: ``True``.
        compute_uv
            If ``True`` then left and right singular vectors will be computed and
            returned in ``U`` and ``Vh``, respectively. Otherwise, only the singular
            values will be computed, which can be significantly faster.
        .. note::
            with backend set as torch, svd with still compute left and right singular
            vectors irrespective of the value of compute_uv, however Ivy will
            still only return the
            singular values.

        Returns
        -------
        .. note::
            once complex numbers are supported, each square matrix must be Hermitian.

        ret
            A container of a namedtuples ``(U, S, Vh)``. More details in ivy.svd.


        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.random_normal(shape = (9, 6))
        >>> y = ivy.random_normal(shape = (2, 4))
        >>> z = ivy.Container(a=x, b=y)
        >>> ret = ivy.Container.static_svd(z)
        >>> aU, aS, aVh = ret.a
        >>> bU, bS, bVh = ret.b
        >>> print(aU.shape, aS.shape, aVh.shape, bU.shape, bS.shape, bVh.shape)
        (9, 9) (6,) (6, 6) (2, 2) (2,) (4, 4)
        """
        return ContainerBase.cont_multi_map_in_function(
            "svd",
            x,
            compute_uv=compute_uv,
            full_matrices=full_matrices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def svd(
        self: ivy.Container,
        /,
        *,
        compute_uv: Union[bool, ivy.Container] = True,
        full_matrices: Union[bool, ivy.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.svd. This method simply
        wraps the function, and so the docstring for ivy.svd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with array leaves having shape ``(..., M, N)`` and whose
            innermost two dimensions form matrices on which to perform singular value
            decomposition. Should have a floating-point data type.
        full_matrices
            If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has
            shape ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``,
            compute on             the leading ``K`` singular vectors, such that ``U``
            has shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where
            ``K = min(M, N)``. Default: ``True``.
        compute_uv
            If ``True`` then left and right singular vectors will be computed and
            returned in ``U`` and ``Vh``, respectively. Otherwise, only the singular
            values will be computed, which can be significantly faster.
        .. note::
            with backend set as torch, svd with still compute left and right singular
            vectors irrespective of the value of compute_uv, however Ivy will
            still only return the
            singular values.

        Returns
        -------
        .. note::
            once complex numbers are supported, each square matrix must be Hermitian.

        ret
            A container of a namedtuples ``(U, S, Vh)``. More details in ivy.svd.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.random_normal(shape = (9, 6))
        >>> y = ivy.random_normal(shape = (2, 4))
        >>> z = ivy.Container(a=x, b=y)
        >>> ret = z.svd()
        >>> print(ret[0], ret[1], ret[2])
        {
            a: (<class ivy.data_classes.array.array.Array> shape=[9, 9]),
            b: ivy.array([[-0.3475602, -0.93765765],
                          [-0.93765765, 0.3475602]])
        } {
            a: ivy.array([3.58776021, 3.10416126, 2.80644298, 1.87024701, 1.48127627,
                          0.79101127]),
            b: ivy.array([1.98288572, 0.68917423])
        } {
            a: (<class ivy.data_classes.array.array.Array> shape=[6, 6]),
            b: (<class ivy.data_classes.array.array.Array> shape=[4, 4])
        }
        """
        return self._static_svd(
            self,
            compute_uv=compute_uv,
            full_matrices=full_matrices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_svdvals(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "svdvals",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def svdvals(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_svdvals(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tensordot(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axes: Union[int, Tuple[List[int], List[int]], ivy.Container] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tensordot",
            x1,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensordot(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axes: Union[int, Tuple[List[int], List[int]], ivy.Container] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_tensordot(
            self,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tensorsolve(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axes: Optional[Union[int, Tuple[List[int], List[int]], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tensorsolve",
            x1,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensorsolve(
        self: ivy.Container,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axes: Optional[Union[int, Tuple[List[int], List[int]], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_tensorsolve(
            self,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_trace(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        offset: Union[int, ivy.Container] = 0,
        axis1: Union[int, ivy.Container] = 0,
        axis2: Union[int, ivy.Container] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.trace. This method
        Returns the sum along the specified diagonals of a matrix (or a stack
        of matrices).

        Parameters
        ----------
        x
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        offset
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative. Defaults to 0.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``0.`` .
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``1.`` .
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the traces and whose shape is determined by removing
            the last two dimensions and storing the traces in the last array dimension.
            For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``,
            then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array must have the same data type as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:
        >>> x = ivy.Container(
        ...    a = ivy.array([[7, 1, 2],
        ...                   [1, 3, 5],
        ...                   [0, 7, 4]]),
        ...    b = ivy.array([[4, 3, 2],
        ...                   [1, 9, 5],
        ...                   [7, 0, 6]])
        )
        >>> y = x.Container.static_trace(x)
        >>> print(y)
        {
            a: ivy.array(14),
            b: ivy.array(19)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "trace",
            x,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trace(
        self: ivy.Container,
        /,
        *,
        offset: Union[int, ivy.Container] = 0,
        axis1: Union[int, ivy.Container] = 0,
        axis2: Union[int, ivy.Container] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.trace. This method
        Returns the sum along the specified diagonals of a matrix (or a stack
        of matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        offset
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative. Defaults to 0.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``0.`` .
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``1.`` .
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the traces and whose shape is determined by removing
            the last two dimensions and storing the traces in the last array dimension.
            For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``,
            then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array must have the same data type as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:
        >>> x = ivy.Container(
        ...    a = ivy.array([[7, 1, 2],
        ...                   [1, 3, 5],
        ...                   [0, 7, 4]]),
        ...    b = ivy.array([[4, 3, 2],
        ...                   [1, 9, 5],
        ...                   [7, 0, 6]]))
        >>> y = x.trace()
        >>> print(y)
        {
            a: ivy.array(14),
            b: ivy.array(19)
        }
        """
        return self._static_trace(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vecdot(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "vecdot",
            x1,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vecdot(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_vecdot(
            self,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vector_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        ord: Union[int, float, Literal[inf, -inf], ivy.Container] = 2,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.vector_norm. This method
        simply wraps the function, and so the docstring for ivy.vector_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array. Should have a floating-point data type.
        axis
            If an integer, ``axis`` specifies the axis (dimension)
            along which to compute vector norms. If an n-tuple,
            ``axis`` specifies the axes (dimensions) along
            which to compute batched vector norms. If ``None``, the
             vector norm must be computed over all array values
             (i.e., equivalent to computing the vector norm of
            a flattened array). Negative indices must be
            supported. Default: ``None``.
        keepdims
            If ``True``, the axes (dimensions) specified by ``axis``
            must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible
            with the input array (see :ref:`broadcasting`). Otherwise,
            if ``False``, the axes (dimensions) specified by ``axis`` must
            not be included in the result. Default: ``False``.
        ord
            order of the norm. The following mathematical norms must be supported:

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

            The following non-mathematical "norms" must be supported:

            +------------------+--------------------------------+
            | ord              | description                    |
            +==================+================================+
            | 0                | sum(a != 0)                    |
            +------------------+--------------------------------+
            | -1               | 1./sum(1./abs(a))              |
            +------------------+--------------------------------+
            | -2               | 1./sqrt(sum(1./abs(a)/*/*2))   | # noqa
            +------------------+--------------------------------+
            | -inf             | min(abs(a))                    |
            +------------------+--------------------------------+
            | (int,float < 1)  | sum(abs(a)/*/*ord)/*/*(1./ord) |
            +------------------+--------------------------------+

            Default: ``2``.
        dtype
            data type that may be used to perform the computation more precisely. The
            input array ``x`` gets cast to ``dtype`` before the function's computations.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the vector norms. If ``axis`` is
            ``None``, the returned array must be a zero-dimensional
            array containing a vector norm. If ``axis`` is
            a scalar value (``int`` or ``float``), the returned array
            must have a rank which is one less than the rank of ``x``.
            If ``axis`` is a ``n``-tuple, the returned array must have
             a rank which is ``n`` less than the rank of ``x``. The returned
            array must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a = [1., 2., 3.], b = [-2., 0., 3.2])
        >>> y = ivy.Container.static_vector_norm(x)
        >>> print(y)
        {
            a: ivy.array([3.7416575]),
            b: ivy.array([3.77359247])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vector_norm",
            x,
            axis=axis,
            keepdims=keepdims,
            ord=ord,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vector_norm(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        ord: Union[int, float, Literal[inf, -inf], ivy.Container] = 2,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        r"""ivy.Container instance method variant of ivy.vector_norm. This
        method simply wraps the function, and so the docstring for
        ivy.vector_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            If an integer, ``axis`` specifies the axis (dimension)
            along which to compute vector norms. If an n-tuple, ``axis``
            specifies the axes (dimensions) along which to compute
            batched vector norms. If ``None``, the vector norm must be
            computed over all array values (i.e., equivalent to computing
            the vector norm of a flattened array). Negative indices must
            be supported. Default: ``None``.
        keepdims
            If ``True``, the axes (dimensions) specified by ``axis`` must
            be included in the result as singleton dimensions, and, accordingly,
            the result must be compatible with the input array
            (see :ref:`broadcasting`).Otherwise, if ``False``, the axes
            (dimensions) specified by ``axis`` must not be included in
            the result. Default: ``False``.
        ord
            order of the norm. The following mathematical norms must be supported:

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

            The following non-mathematical "norms" must be supported:

            +------------------+--------------------------------+
            | ord              | description                    |
            +==================+================================+
            | 0                | sum(a != 0)                    |
            +------------------+--------------------------------+
            | -1               | 1./sum(1./abs(a))              |
            +------------------+--------------------------------+
            | -2               | 1./sqrt(sum(1./abs(a)/*/*2))   | # noqa
            +------------------+--------------------------------+
            | -inf             | min(abs(a))                    |
            +------------------+--------------------------------+
            | (int,float < 1)  | sum(abs(a)/*/*ord)/*/*(1./ord) |
            +------------------+--------------------------------+

            Default: ``2``.
        dtype
            data type that may be used to perform the computation more precisely. The
            input array ``x`` gets cast to ``dtype`` before the function's computations.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the vector norms. If ``axis`` is ``None``,
            the returned array must be a zero-dimensional array containing
            a vector norm. If ``axis`` is a scalar value (``int`` or ``float``),
            the returned array must have a rank which is one less than the
            rank of ``x``. If ``axis`` is a ``n``-tuple, the returned
            array must have a rank which is ``n`` less than the rank of
            ``x``. The returned array must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a = [1., 2., 3.], b = [-2., 0., 3.2])
        >>> y = x.vector_norm()
        >>> print(y)
        {
            a: ivy.array([3.7416575]),
            b: ivy.array([3.77359247])
        }
        """
        return self._static_vector_norm(
            self,
            axis=axis,
            keepdims=keepdims,
            ord=ord,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vector_to_skew_symmetric_matrix(
        vector: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "vector_to_skew_symmetric_matrix",
            vector,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vector_to_skew_symmetric_matrix(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_vector_to_skew_symmetric_matrix(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vander(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        N: Optional[Union[int, ivy.Container]] = None,
        increasing: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.vander. This method
        simply wraps the function, and so the docstring for ivy.vander also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            ivy container that contains 1-D arrays.
        N
            Number of columns in the output. If N is not specified,
            a square array is returned (N = len(x))
        increasing
            Order of the powers of the columns. If True, the powers increase
            from left to right, if False (the default) they are reversed.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container that contains the Vandermonde matrix of the arrays included
            in the input container.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(
                a = ivy.array([1, 2, 3, 5])
                b = ivy.array([6, 7, 8, 9])
            )
        >>> ivy.Container.static_vander(x)
        {
            a: ivy.array(
                    [[  1,   1,   1,   1],
                    [  8,   4,   2,   1],
                    [ 27,   9,   3,   1],
                    [125,  25,   5,   1]]
                    ),
            b: ivy.array(
                    [[216,  36,   6,   1],
                    [343,  49,   7,   1],
                    [512,  64,   8,   1],
                    [729,  81,   9,   1]]
                    )
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vander",
            x,
            N=N,
            increasing=increasing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vander(
        self: ivy.Container,
        /,
        *,
        N: Optional[Union[int, ivy.Container]] = None,
        increasing: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.vander. This method
        Returns the Vandermonde matrix of the input array.

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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            an container containing the Vandermonde matrices of the arrays
            included in the input container.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(
                a = ivy.array([1, 2, 3, 5])
                b = ivy.array([6, 7, 8, 9])
            )
        >>> x.vander()
        {
            a: ivy.array(
                    [[  1,   1,   1,   1],
                    [  8,   4,   2,   1],
                    [ 27,   9,   3,   1],
                    [125,  25,   5,   1]]
                    ),
            b: ivy.array(
                    [[216,  36,   6,   1],
                    [343,  49,   7,   1],
                    [512,  64,   8,   1],
                    [729,  81,   9,   1]]
                    )
        }
        """
        return self._static_vander(
            self,
            N=N,
            increasing=increasing,
            out=out,
        )

    @staticmethod
    def static_general_inner_product(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        n_modes: Optional[Union[int, ivy.Container]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.general_inner_product.
        This method simply wraps the function, and so the docstring for
        ivy.general_inner_product also applies to this method with minimal
        changes.

        Parameters
        ----------
        x1
            First input container containing input array.
        x2
            First input container containing input array.
        n_modes
            int, default is None. If None, the traditional inner product is returned
            (i.e. a float) otherwise, the product between the `n_modes` last modes of
            `x1` and the `n_modes` first modes of `x2` is returned. The resulting
            tensor's order is `len(x1) - n_modes`.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            Alternate output container in which to place the result.
            The default is None.

        Returns
        -------
        ret
            Container including the inner product tensor.

        Examples
        --------
        >>> x = ivy.Container(
                a=ivy.reshape(ivy.arange(4), (2, 2)),
                b=ivy.reshape(ivy.arange(8), (2, 4)),
            )
        >>> ivy.Container.general_inner_product(x, 1)
            {
                a: ivy.array(6),
                b: ivy.array(28)
            }
        """
        return ContainerBase.cont_multi_map_in_function(
            "general_inner_product",
            x1,
            x2,
            n_modes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def general_inner_product(
        self: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        n_modes: Optional[Union[int, ivy.Container]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.general_inner_product.

        This method simply wraps the function, and so the docstring for
        ivy.general_inner_product also applies to this method with
        minimal changes.
        """
        return self.static_general_inner_product(
            self,
            x2,
            n_modes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
