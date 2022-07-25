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

    def cross(
            self: ivy.Container,
            x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            axis: int = -1,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_nests: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.cross(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.det(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.diagonal(x_, offset, axis1, axis2)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
            out=None
        )

    def eigvalsh(
            self: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.eigvalsh(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.inv(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.matrix_norm(x_, ord, keepdims)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.matrix_power(x_, n) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.matrix_rank(x_, rtol) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.matrix_rank(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def outer(
            self: ivy.Container,
            x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_nests: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.outer(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
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
    ) -> NamedTuple:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.qr(x_, mode) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.slogdet(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def solve(
            self: ivy.Container,
            x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_nests: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.solve(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
            out=out,
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.svd(x_, full_matrices) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
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
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.svdvals(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def tensordot(
            self: ivy.Container,
            x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            axes: Union[int, Tuple[List[int], List[int]]] = 2,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_nests: bool = False,
            *,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"x1": self}
        if ivy.is_array(x2):
            kw["x2"] = x2
        else:
            conts["x2"] = x2
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.tensordot(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
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
    ):
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
