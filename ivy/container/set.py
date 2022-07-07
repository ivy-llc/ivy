# local
from ivy.container.base import ContainerBase
import ivy
from typing import Optional, List, Union, Dict

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSet(ContainerBase):

    @staticmethod
    def static_unique_all(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "unique_all",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def unique_all(
            self: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return self.static_unique_all(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    @staticmethod
    def static_unique_counts(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "unique_counts",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def unique_counts(
            self: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return self.static_unique_counts(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    @staticmethod
    def static_unique_values(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            *,
            out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "unique_values",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )

    def unique_values(
            self: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return self.static_unique_values(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    @staticmethod
    def static_unique_inverse(
            x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "unique_inverse",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )

    def unique_inverse(
            self: ivy.Container,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False
    ) -> ivy.Container:
        return self.static_unique_inverse(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences
        )
