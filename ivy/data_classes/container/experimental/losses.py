# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLossesExperimental(ContainerBase):
    @staticmethod
    def _static_hinge_embedding_loss(
        input: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        target: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        margin: Optional[Union[float, ivy.Container]] = 1.0,
        reduction: Optional[Union[str, ivy.Container]] = "mean",  # Added parameter
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hinge_embedding_loss. This method
        simply wraps the function, and so the docstring for ivy.hinge_embedding_loss
        also applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing targeticted scores or logits.
        margin
            Margin value for the hinge loss. Default: 1.0.
        reduction
            Reduction method to be applied to the output. Default: 'mean'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The hinge embedding loss between the given targetictions and input labels.

        Examples
        --------
        # ... (example code)
        """
        return ContainerBase.cont_multi_map_in_function(
            "hinge_embedding_loss",
            input,
            target,
            margin=margin,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hinge_embedding_loss(
        self: ivy.Container,
        target: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        margin: Optional[Union[float, ivy.Container]] = 1.0,
        reduction: Optional[Union[str, ivy.Container]] = "mean",  # Added parameter
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hinge_embedding_loss. This method
        simply wraps the function, and so the docstring for ivy.hinge_embedding_loss
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container containing input labels.
        target
            input array or container containing targeticted scores or logits.
        margin
            Margin value for the hinge loss. Default: 1.0.
        reduction
            Reduction method to be applied to the output. Default: 'mean'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The hinge embedding loss between the given targetictions and input labels.

        Examples
        --------
        # ... (example code)
        """
        return self._static_hinge_embedding_loss(
            self,
            target,
            margin=margin,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
