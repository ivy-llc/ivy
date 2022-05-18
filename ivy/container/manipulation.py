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
            ),
            out,
        )

    def expand_dims(
        self: ivy.Container,
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.expand_dims(x_, axis=axis)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def permute_dims(
        self: ivy.Container,
        axes: Tuple[int, ...],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.permute_dims(x_, axes=axes)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def flip(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.flip(x_, axis=axis) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def reshape(
        self: ivy.Container,
        shape: Tuple[int, ...],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.reshape(x_, shape=shape) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def roll(
        self: ivy.Container,
        shift: Union[int, Tuple[int, ...]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.roll(x_, shift=shift, axis=axis)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def squeeze(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.squeeze(x_, axis=axis) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def stack(
        self: ivy.Container,
        x: Union[
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
        for y in x:
            if ivy.is_ivy_container(y):
                conts.append(y)
                arrays.append(None)
            else:
                arrays.append(y)
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs_, _: ivy.stack(
                    x=[a if ivy.exists(a) else xs_.pop(0) for a in arrays], axis=axis
                )
                if ivy.is_array(xs_[0])
                else xs_,
                conts,
                key_chains,
                to_apply,
                prune_unapplied,
            ),
            out,
        )
