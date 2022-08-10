# global
from typing import Optional, Union, List, Dict, Tuple

# local
from ivy.container.base import ContainerBase
import ivy

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
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.cross. This method simply wraps the
            function, and so the docstring for ivy.cross also applies to this method
            with minimal changes.

        Parameters
        ----------
        x1
            first input array. Should have a numeric data type.
        x2
            second input array. Must have the same shape as x1. Should have a numeric data
            type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to compute
            the cross product.vIf set to -1, the function computes the cross product for
            vectors defined by the last axis (dimension). Default: -1.
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
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a container containing the cross products. The returned array must have a data
            type determined by Type Promotion Rules.

        Container Static Method Examples
        --------

        With :code:`ivy.Container` inputs:
        >>> x1 = ivy.Container.static_cross(a=ivy.array([5., 0., 0.]), b=ivy.array([0., 0., 2.]))
        >>> x2 = ivy.Container.static_cross(a=ivy.array([0., 7., 0.]), b=ivy.array([3., 0., 0.]))
        >>> z = ivy.Container.static_cross(x1,x2)
        >>> print(z)
        {
        a: ivy.array([0., 0., 35.]),
        b: ivy.array([0., 6., 0.])
        }

        With a combination of :code:`ivy.Array`
        and :code:`ivy.Container` inputs:

        >>> x1 = ivy.array([9., 0., 3.])
        >>> x2 = ivy.Container.static_cross(a=ivy.array([1., 1., 0.]), b=ivy.array([1., 0., 1.]))
        >>> z = x1.cross(x1,x2)
        >>> print(z)
        {
        a: ivy.array([-3., 3., 9.]),
        b: ivy.array([0., -6., 0.])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "cross",
            x1,
            x2,
            axis,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    def cross(
        self: ivy.Container,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.cross. This method simply wraps the
            function, and so the docstring for ivy.cross also applies to this method
            with minimal changes.

        Parameters
        ----------
        x1
            container with first input array. Should have a numeric data type.
        x2
            second input array. Must have the same shape as x1. Should have a numeric data
            type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to compute
            the cross product.vIf set to -1, the function computes the cross product for
            vectors defined by the last axis (dimension). Default: -1.
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
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a container containing the cross products. The returned array must have a data
            type determined by Type Promotion Rules.

        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_.

        Instance Method Examples
        ------------------------

        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([5., 0., 0.]), b=ivy.array([0., 0., 2.]))
        >>> x2 = ivy.Container(a=ivy.array([0., 7., 0.]), b=ivy.array([3., 0., 0.]))
        >>> z = x1.cross(x1, x2)
        >>> print(z)
        {
        a: ivy.array([0., 0., 35.]),
        b: ivy.array([0., 6., 0.])
        }

        With a combination of :code:`ivy.Array`
        and :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1., 1., 0.]), b=ivy.array([1., 0., 1.]))
        >>> y = ivy.array([9., 0., 3.])
        >>> z = ivy.cross(x,y)
        >>> print(z)
        {
        a: ivy.array([3., -3., 9.]),
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
