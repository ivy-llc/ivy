# global
from typing import Optional, Union, List, Dict, Sequence

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithTensors(ContainerBase):
    @staticmethod
    def static_unfold(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        mode: Optional[int] = 0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.unfold. This method simply wraps the
        function, and so the docstring for ivy.unfold also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            input container
        mode
            indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

        Returns
        -------
        ret
            container of unfolded tensors
        """
        return ContainerBase.cont_multi_map_in_function(
            "unfold",
            input,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def unfold(
        self: ivy.Container,
        /,
        mode: Optional[int] = 0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.unfold. This method simply wraps
        the function, and so the docstring for ivy.unfold also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container

        Returns
        -------
        ret
            container of vectorised tensors
        """
        return self.static_unfold(
            self,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fold(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        mode: int,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fold. This method simply wraps the
        function, and so the docstring for ivy.fold also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            input container
        mode
            the mode of the unfolding
        shape
            shape of the original tensors before unfolding

        Returns
        -------
        ret
            container of folded tensors
        """
        return ContainerBase.cont_multi_map_in_function(
            "fold",
            input,
            mode,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fold(
        self: ivy.Container,
        /,
        mode: int,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fold. This method simply wraps the
        function, and so the docstring for ivy.fold also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container
        mode
            the mode of the unfolding
        shape
            shape of the original tensors before unfolding
        Returns
        -------
        ret
            container of vectorised tensors
        """
        return self.static_fold(
            self,
            mode=mode,
            shape=shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
