# local
from typing import Union, Literal, Optional, List, Dict

import ivy

# from ivy import DevDistItem
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# Placeholder for type hints.
class DevDistItem:
    pass


# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev_unify_array(
        xs: DevDistItem,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        device: Union[ivy.Device, ivy.NativeDevice],
        mode: Literal["concat", "mean", "sum"],
        axis: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "dev_unify_array",
            xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            device=device,
            mode=mode,
            axis=axis,
        )
