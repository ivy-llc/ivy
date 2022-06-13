# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithLosses(ContainerBase):
    def cross_entropy(
        self: ivy.Container,
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Optional[int] = -1,
        epsilon: Optional[float] = 1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        kw = {}
        conts = {"true": self}
        if ivy.is_array(pred):
            kw["pred"] = pred
        else:
            conts["pred"] = pred
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.cross_entropy(
                    **dict(zip(conts.keys(), xs)), **kw, axis=axis, epsilon=epsilon
                )
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
            ),
            out,
        )

    def binary_cross_entropy(
        self: ivy.Container,
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        epsilon: Optional[float] = 1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"true": self}
        if ivy.is_array(pred):
            kw["pred"] = pred
        else:
            conts["pred"] = pred
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.binary_cross_entropy(
                    **dict(zip(conts.keys(), xs)), **kw, epsilon=epsilon
                )
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
            ),
            out,
        )

    def sparse_cross_entropy(
        self: ivy.Container,
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Optional[int] = -1,
        epsilon: Optional[float] = 1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        kw = {}
        conts = {"true": self}
        if ivy.is_array(pred):
            kw["pred"] = pred
        else:
            conts["pred"] = pred
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs, _: ivy.sparse_cross_entropy(
                    **dict(zip(conts.keys(), xs)), **kw, axis=axis, epsilon=epsilon
                )
                if ivy.is_array(xs[0])
                else xs,
                list(conts.values()),
                key_chains,
                to_apply,
                prune_unapplied,
            ),
            out,
        )

    pass
