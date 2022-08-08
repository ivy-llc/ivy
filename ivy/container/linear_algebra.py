# global
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Dict

# local
from ivy.container.base import ContainerBase
import ivy

inf = float("inf")

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor,PyMethodParameters
class ContainerWithLinearAlgebra(ContainerBase):
    def matmul(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_nests: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        cont_keys = conts.keys()
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.matmul(**dict(zip(cont_keys, xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
            out,
        )

    @staticmethod
    def static_cholesky(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        upper: Union[int, Tuple[int, ...], ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cholesky. This method simply wraps
        the function, and so the docstring for ivy.cholesky also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input array or container having shape (..., M, M) and whose innermost two
            dimensions form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        With one :code:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[3., -1.], [-1., 3.]]), \
                              b=ivy.array([[2., 1.], [1., 1.]]))
        >>> y = ivy.Container.static_cholesky(x, 'false')
        >>> print(y)
        {
            a: ivy.array([[1.73, -0.577], 
                            [0., 1.63]]),
            b: ivy.array([[1.41, 0.707], 
                            [0., 0.707]])
         }
        With multiple :code:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.array([[3., -1], [-1., 3.]]), \
                              b=ivy.array([[2., 1.], [1., 1.]]))
        >>> upper = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, upper)
        >>> print(y)
        {
            a: ivy.array([[3., 3.], 
                         [-1., -1.]]),
            b: ivy.array([[1., 1.], 
                          [1., 2.]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "cholesky",
            x,
            upper,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cholesky(
        self: ivy.Container,
        upper: Union[int, Tuple[int, ...], ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cholesky. This method simply wraps
        the function, and so the docstring for ivy.cholesky also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container having shape (..., M, M) and whose innermost two dimensions
            form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        >>> x = ivy.Container(a=ivy.array([[3., -1],[-1., 3.]]), \
                              b=ivy.array([[2., 1.],[1., 1.]]))
        >>> y = x.cholesky('false')
        >>> print(y)
        {
            a: ivy.array([[1.73, -0.577],
                            [0., 1.63]]),
            b: ivy.array([[1.41, 0.707],
                            [0., 0.707]])
        }
        """
        return self.static_cholesky(
            self,
            upper,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_cross(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cross.
        This method simply wraps the function, and so the docstring
        for ivy.cross also applies to this method with minimal changes.

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
            Default: -1.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        With one :code:`ivy.Container` input:

        >>> x = ivy.array([9., 0., 3.])
        >>> y = ivy.Container(a=ivy.array([1., 1., 0.]), b=ivy.array([1., 0., 1.]))
        >>> z = ivy.Container.static_cross(x, y)
        >>> print(z)
        {
            a: ivy.array([-3., 3., 9.]),
            b: ivy.array([0., -6., 0.])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = x = ivy.Container(a=ivy.array([5., 0., 0.]), b=ivy.array([0., 0., 2.]))
        >>> y = ivy.Container(a=ivy.array([0., 7., 0.]), b=ivy.array([3., 0., 0.]))
        >>> z = ivy.Container.static_cross(x, y)
        >>> print(z)
        {
            a: ivy.array([0., 0., 35.]),
            b: ivy.array([0., 6., 0.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "cross",
            x1,
            x2,
            axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cross(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cross.
        This method simply wraps the function, and so the docstring
        for ivy.cross also applies to this method with minimal changes.

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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        return self.static_cross(
            self,
            x2,
            axis,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_det(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_det(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_diagonal(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "diagonal",
            x,
            offset,
            axis1,
            axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diagonal(
        self: ivy.Container,
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_diagonal(
            self,
            offset,
            axis1,
            axis2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    def eigh(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> NamedTuple:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.eigh(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=None,
        )

    @staticmethod
    def static_inv(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "inv",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def inv(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_inv(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_matrix_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.matrix_norm.
        This method simply wraps the function, and so the docstring for
        ivy.matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array.
        ord
            Order of the norm. Default is 2.
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.
        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        """
        return ContainerBase.multi_map_in_static_method(
            "matrix_norm",
            x,
            ord,
            keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_norm(
        self: ivy.Container,
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.matrix_norm.
        This method simply wraps the function, and so the docstring for
        ivy.matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        ord
            Order of the norm. Default is 2.
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.
        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        """
        return self.static_matrix_norm(
            self,
            ord,
            keepdims,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_matrix_power(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        n: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        n: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_matrix_power(
            self,
            n,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_matrix_rank(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        rtol: Optional[Union[float, Tuple[float]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.matrix_rank.
        This method returns the rank (i.e., number of non-zero singular values)
        of a matrix (or a stack of matrices).

        Parameters
        ----------        
        x
            input array or container having shape ``(..., M, N)`` and whose innermost
            two dimensions form ``MxN`` matrices. Should have a floating-point data 
            type.
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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        With :code: `ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[1., 0.], [0., 1.]]), \
                              b=ivy.array([[1., 0.], [0., 0.]]))
        >>> y = ivy.Container.static_matrix_rank(x)
        >>> print(y)
        {
            a: ivy.array(2.),
            b: ivy.array(1.)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "matrix_rank",
            x,
            rtol,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_rank(
        self: ivy.Container,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.matrix_rank.
        This method returns the rank (i.e., number of non-zero singular values)
        of a matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two 
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
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
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.        
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        With :code: `ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[1., 0.], [0., 1.]]), \
                                b=ivy.array([[1., 0.], [0., 0.]]))
        >>> y = x.matrix_rank()
        >>> print(y)
        {
            a: ivy.array(2),
            b: ivy.array(1)
        }
        """
        return self.static_matrix_rank(
            self,
            rtol,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_matrix_transpose(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "matrix_transpose",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_transpose(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_matrix_transpose(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_outer(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_outer(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_qr(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mode: str = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "qr",
            x,
            mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def qr(
        self: ivy.Container,
        mode: str = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_qr(
            self,
            mode,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_slogdet(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "slogdet",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def slogdet(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_slogdet(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_solve(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "solve",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def solve(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_solve(
            self,
            x2,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_svd(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        full_matrices: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> Union[ivy.Container, Tuple[ivy.Container, ...]]:
        return ContainerBase.multi_map_in_static_method(
            "svd",
            x,
            full_matrices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    # Unsure
    def svd(
        self: ivy.Container,
        full_matrices: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> Union[ivy.Container, Tuple[ivy.Container, ...]]:
        return self.static_svd(
            self,
            full_matrices,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_svdvals(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_svdvals(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_tensordot(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "tensordot",
            x1,
            x2,
            axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensordot(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axes: Union[int, Tuple[List[int], List[int]]] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_tensordot(
            self,
            x2,
            axes,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_trace(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        offset: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "trace",
            x,
            offset,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trace(
        self: ivy.Container,
        offset: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_trace(
            self,
            offset,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_vecdot(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "vecdot",
            x1,
            x2,
            axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vecdot(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_vecdot(
            self,
            x2,
            axis,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_vector_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        ord: Union[int, float, Literal[inf, -inf]] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        r"""
        ivy.Container static method variant of ivy.vector_norm.
        This method simply wraps the function, and so the docstring for
        ivy.vector_norm also applies to this method with minimal changes.

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

        """
        return ContainerBase.multi_map_in_static_method(
            "vector_norm",
            x,
            axis,
            keepdims,
            ord,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vector_norm(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: bool = False,
        ord: Union[int, float, Literal[inf, -inf]] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        r"""
        ivy.Container instance method variant of ivy.vector_norm.
        This method simply wraps the function, and so the docstring for
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

        """
        return self.static_vector_norm(
            self,
            axis,
            keepdims,
            ord,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_vector_to_skew_symmetric_matrix(
        vector: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_vector_to_skew_symmetric_matrix(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
