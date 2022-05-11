# global
from typing import Optional, Union, List, Tuple, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithManipulation(ContainerBase):
    def concat(
        self: ivy.Container,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        conts = [self]
        arrays = [None]
        for x in xs:
            if ivy.is_ivy_container(x):
                conts.append(x)
                arrays.append(None)
            else:
                arrays.append(x)
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs_, _: ivy.concat(
                    xs=[a if ivy.exists(a) else xs_.pop(0) for a in arrays], axis=axis
                )
                if ivy.is_array(xs_[0])
                else xs_,
                conts,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )
