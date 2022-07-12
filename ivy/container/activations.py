# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithActivations(ContainerBase):
    def softplus(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.softplus. 
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.
        """
        return ContainerWithActivations.static_softplus(
            self, key_chains, to_apply, prune_unapplied, map_sequences, out=out
        )
    
    @staticmethod
    def static_softplus(
        x: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.softplus.
        This method simply wraps the function, and so the docstring
        for ivy.softplus also applies to this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "softplus",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
