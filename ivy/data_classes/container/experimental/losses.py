# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLosses(ContainerBase):
    @staticmethod
    def _static_l1_loss(
        true: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Union[str, ivy.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.l1_loss. This method simply wraps the
        function, and so the docstring for ivy.l1_loss also applies to this method with
        minimal changes.

        Parameters
        ----------
        true
            input array or container.
        pred
            input array or container containing the predicted values.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
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
            The L1 loss between the input array and the predicted values.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 2, 2]), b=ivy.array([5, 5, 5]))
        >>> z = ivy.Container.static_l1_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(0.)
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.Container(a=ivy.array([2, 2, 2]), b=ivy.array([5, 5, 5]))
        >>> z = ivy.Container.static_l1_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(4.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "l1_loss",
            true,
            pred,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l1_loss(
        self: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Union[str, ivy.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.l1_loss. This method simply wraps
        the function, and so the docstring for ivy.l1_loss also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        pred
            input array or container containing the predicted values.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
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
            The L1 loss between the input array and the predicted values.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 2, 2]), b=ivy.array([5, 5, 5]))
        >>> z = x.l1_loss(y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(0.)
        }
        """
        return self._static_l1_loss(
            self,
            pred,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
