# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor,PyMethodParameters
class ContainerWithLinearAlgebra(ContainerBase):
    def matmul(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
                lambda xs, _: ivy.matmul(**dict(zip(conts.keys(), xs)), **kw)
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
            ),
            out,
        )

    def outer(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            ),
            out,
        )
