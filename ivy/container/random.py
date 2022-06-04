# global
from typing import Optional, Union, List, Dict, Tuple

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithRandom(ContainerBase):
    @staticmethod
    def static_random_uniform(
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "random_uniform",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_uniform(
        self: ivy.Container,
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_random_uniform(
            low,
            high,
            self,
            device,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
