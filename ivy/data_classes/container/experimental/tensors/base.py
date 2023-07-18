# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithTensors(ContainerBase):
    @staticmethod
    def static_tensor_to_vec(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tensor_to_vec. This method simply
        wraps the function, and so the docstring for ivy.tensor_to_vec also applies to
        this method with minimal changes.

        Parameters
        ----------
        input
            input container

        Returns
        -------
        ret
            container of vectorised tensors
        """
        return ContainerBase.cont_multi_map_in_function(
            "tensor_to_vec",
            input,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensor_to_vec(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tensor_to_vec. This method simply
        wraps the function, and so the docstring for ivy.tensor_to_vec also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container

        Returns
        -------
        ret
            container of vectorised tensors
        """
        return self.static_tensor_to_vec(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
