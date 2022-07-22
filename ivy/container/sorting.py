# local
from ivy.container.base import ContainerBase
import ivy
from typing import Optional, List, Union, Dict


# noinspection PyMissingConstructor
class ContainerWithSorting(ContainerBase):
    @staticmethod
    def static_argsort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "argsort",
            x,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def argsort(
        self: ivy.Container,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return self.static_argsort(
            self,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "sort",
            x,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sort(
        self: ivy.Container,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return self.static_sort(
            self,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
