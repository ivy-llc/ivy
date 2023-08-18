from typing import Optional, Union, List, Dict
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLossesExperimental(ContainerBase):
    @staticmethod
    def _static_gaussian_nll_loss(
        input: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        variance: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        target: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Union[str, ivy.Container] = "mean",
        epsilon: Union[float, ivy.Container] = 1e-7,
        full: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.gaussian_nll_loss. This method simply
        wraps the function, and so the docstring for ivy.gaussian_nll_loss also applies
        to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container containing the inputs of the
             Gaussian distributions.
        variance
            input array or container containing the variances
            of the Gaussian distributions.
        target
            input array or container containing the true values.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'sum'``.
        epsilon
            a float specifying the amount of smoothing when calculating the loss.
            Default: ``1e-7``.
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
            The Gaussian negative log likelihood loss between the given distributions.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> input = ivy.Container(a=ivy.array([0.0, 1.0, -1.0]))
        >>> variance = ivy.Container(a=ivy.array([0.0, 0.0, 0.0]))
        >>> target = ivy.Container(a=ivy.array([1.0, 1.0, -1.0]))
        >>> loss = ivy.Container.static_gaussian_nll_loss(input, variance, target)
        >>> print(loss)
        {
            a: ivy.array(0.5)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "gaussian_nll_loss",
            input,
            variance,
            target,
            full=full,
            reduction=reduction,
            epsilon=epsilon,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gaussian_nll_loss(
        self: ivy.Container,
        variance: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        target: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Union[str, ivy.Container] = "mean",
        full: Union[bool, ivy.Container] = False,
        epsilon: Union[float, ivy.Container] = 1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gaussian_nll_loss. This method
        simply wraps the function, and so the docstring for ivy.gaussian_nll_loss also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container containing the inputs of the Gaussian distributions.
        variance
            input array or container containing the log-variances
            of the Gaussian distributions.
        target
            input array or container containing the true values.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'input'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'sum'``.
        epsilon
            a float specifying the amount of smoothing when calculating the loss.
            Default: ``1e-7``.
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
            The Gaussian negative log likelihood loss between the given distributions.

        Examples
        --------
        >>> input = ivy.Container(a=ivy.array([0.0, 1.0, -1.0]))
        >>> variance = ivy.Container(a=ivy.array([0.0, 0.0, 0.0]))
        >>> target = ivy.Container(a=ivy.array([1.0, 1.0, -1.0]))
        >>> loss = input.gaussian_nll_loss(variance, target)
        >>> print(loss)
        {
            a: ivy.array(0.5)
        }
        """
        return self._static_gaussian_nll_loss(
            self,
            variance,
            target,
            reduction=reduction,
            full=full,
            epsilon=epsilon,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
