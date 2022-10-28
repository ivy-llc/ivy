# global
from typing import Optional, Tuple, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithCreationExtensions(ContainerBase):
    @staticmethod
    def static_triu_indices(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "triu_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )

    def triu_indices(
        self: ivy.Container,
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return self.static_triu_indices(
            self,
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )
