# global
from typing import Optional, Union, Dict, Sequence

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithUtility(ContainerBase):
    @staticmethod
    def static_all(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "all",
            x,
            axis,
            keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def all(
        self: ivy.Container,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_all(
            self,
            axis,
            keepdims,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_any(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "any",
            x,
            axis,
            keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def any(
        self: ivy.Container,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_any(
            self,
            axis,
            keepdims,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
