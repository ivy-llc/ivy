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
        ivy.Container static method variant of ivy.cholesky.
        This method simply wraps the function, and so the docstring
        for ivy.cholesky also applies to this method
        with minimal changes.

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
        ivy.Container instance method variant of ivy.cholesky.
        This method simply wraps the function, and so the docstring
        for ivy.cholesky also applies to this method
        with minimal changes.

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
    def static_matrix_rank(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        """
        ivy.Container static method variant of ivy.matrix_rank.
        This method returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of
        matrices).

        Parameters
        ----------
        x
            input array or container having shape ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices. Should have a floating-point data type.
            
        rtol
            relative tolerance for small singular values. Singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero. If a
            ``float``, the value is equivalent to a zero-dimensional array having a
            floating-point data type determined by :ref:`type-promotion` (as applied to
            ``x``) and must be broadcast against each matrix. If an ``array``, must have a
            floating-point data type and must be compatible with ``shape(x)[:-2]`` (see
            :ref:`broadcasting`). If ``None``, the default value is ``max(M, N) * eps``,
            where ``eps`` must be the machine epsilon associated with the floating-point
            data type determined by :ref:`type-promotion` (as applied to ``x``).
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
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a floating-point
            data type determined by :ref:`type-promotion` and must have shape ``(...)``
            (i.e., must have a shape equal to ``shape(x)[:-2]``).

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
        rtol: Optional[Union[float, Tuple[float], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:

        """
        ivy.Container instance method variant of ivy.matrix_rank.
        This method returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of
        matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices. Should have a floating-point data type.

        rtol
            relative tolerance for small singular values. Singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero. If a
            ``float``, the value is equivalent to a zero-dimensional array having a
            floating-point data type determined by :ref:`type-promotion` (as applied to
            ``x``) and must be broadcast against each matrix. If an ``array``, must have a
            floating-point data type and must be compatible with ``shape(x)[:-2]`` (see
            :ref:`broadcasting`). If ``None``, the default value is ``max(M, N) * eps``,
            where ``eps`` must be the machine epsilon associated with the floating-point
            data type determined by :ref:`type-promotion` (as applied to ``x``).
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
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a floating-point
            data type determined by :ref:`type-promotion` and must have shape ``(...)``
            (i.e., must have a shape equal to ``shape(x)[:-2]``).
            
        Examples
        --------
        With :code: `ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[1., 0.], [0., 1.]]), \
                                b=ivy.array([[1., 0.], [0., 0.]]))
        >>> y = x.matrix_rank()
        >>> print(y)
        {
            a: ivy.array(2.),
            b: ivy.array(1.)
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