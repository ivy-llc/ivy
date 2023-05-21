from ivy.data_classes.container.base import ContainerBase
from typing import Union, Optional, List, Dict
import ivy


class _ContainerWithLossesExperimental(ContainerBase):
    @staticmethod
    def static_binary_cross_entropy_with_logits(
        true: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        epsilon: float = 1e-7,
        pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        reduction: str = "none",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.binary_cross_entropy_with_logits.
        This method simply wraps the function, and so the docstring for
        ivy.binary_cross_entropy_with_logits also applies to this method with minimal
        changes.

        Parameters
        ----------
        true
            input container of true labels.
        pred
            input container of predicted labels as logits.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing
            when calculating the loss. If epsilon is ``0``, no smoothing
            will be applied. Default: ``1e-7``.
        pos_weight
            a weight for positive examples. Must be an array with length equal
            to the number of classes.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The binary cross entropy with logits loss between the given distributions.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 1, 0]),b=ivy.array([0, 0, 1]))
        >>> y = ivy.Container(a=ivy.array([3.6, 1.2, 5.3]),b=ivy.array([1.8, 2.2, 1.2]))
        >>> z = ivy.Container.static_binary_cross_entropy_with_logits(x, y)
        >>> print(z)
        {
            a: ivy.array([0.027, 0.263, 5.305]),
            b: ivy.array([1.953, 2.305, 0.263])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "binary_cross_entropy_with_logits",
            true,
            pred,
            epsilon=epsilon,
            pos_weight=pos_weight,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def binary_cross_entropy_with_logits(
        self: Union[ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        epsilon: float = 1e-7,
        pos_weight: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        reduction: str = "none",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container method variant of ivy.binary_cross_entropy_with_logits. This
        method simply wraps the function, and so the docstring for
        ivy.binary_cross_entropy_with_logits also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            input container of true labels.
        pred
            input container of predicted labels as logits.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing
            when calculating the loss. If epsilon is ``0``, no smoothing
            will be applied. Default: ``1e-7``.
        pos_weight
            a weight for positive examples. Must be an array with length
            equal to the number of classes.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The binary cross entropy with logits loss between the given distributions.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 1, 0]),b=ivy.array([0, 0, 1]))
        >>> y = ivy.Container(a=ivy.array([3.6, 1.2, 5.3]),b=ivy.array([1.8, 2.2, 1.2]))
        >>> z = x.binary_cross_entropy_with_logits(y)
        >>> print(z)
        {
            a: ivy.array([0.027, 0.263, 5.305]),
            b: ivy.array([1.953, 2.305, 0.263])
        }
        """
        return self.static_binary_cross_entropy_with_logits(
            self,
            pred,
            epsilon=epsilon,
            pos_weight=pos_weight,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
