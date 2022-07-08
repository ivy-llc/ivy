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
