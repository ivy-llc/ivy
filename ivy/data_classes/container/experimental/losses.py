# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithLossesExperimental(ContainerBase):
    @staticmethod
    def _static_mse_loss(
        true: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.mse_loss. This method simply wraps
        the function, and so the docstring for ivy.mse_loss also applies to this method
        with minimal changes.

        Parameters
        ----------
        true
            input array or container containing true labels.
        pred
            input array or container containing predicted labels.
        axis
            the axis along which to compute the mean squared error. If axis is ``-1``,
            the mean squared error will be computed along the last dimension.
            Default: ``-1``.
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
            The mean squared error between the true and predicted values.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_mse_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(1.)
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_mse_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(9.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "mse_loss",
            true,
            pred,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def mse_loss(
        self,
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container method variant of ivy.mse_loss. This method simply wraps
        the function, and so the docstring for ivy.mse_loss also applies to this method
        with minimal changes.

        Parameters
        ----------
        pred
            input array or container containing predicted labels.
        axis
            the axis along which to compute the mean squared error. If axis is ``-1``,
            the mean squared error will be computed along the last dimension.
            Default: ``-1``.
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
            The mean squared error between the true and predicted values.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> x.mse_loss(y)
        {
            a: ivy.array(1.),
            b: ivy.array(1.)
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> x.mse_loss(y)
        {
            a: ivy.array(1.),
            b: ivy.array(9.)
        }
        """
        return _ContainerWithLossesExperimental._static_mse_loss(
            pred,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_mae_loss(
        true: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.mae_loss. This method simply wraps
        the function, and so the docstring for ivy.mae_loss also applies to this method
        with minimal changes.

        Parameters
        ----------
        true
            input array or container containing true labels.
        pred
            input array or container containing predicted labels.
        axis
            the axis along which to compute the mean absolute error. If axis is ``-1``,
            the mean absolute error will be computed along the last dimension.
            Default: ``-1``.
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
            The mean absolute error between the true and predicted values.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_mae_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(1.)
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_mae_loss(x, y)
        >>> print(z)
        {
            a: ivy.array(1.),
            b: ivy.array(2.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "mae_loss",
            true,
            pred,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def mae_loss(
        self,
        pred: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container method variant of ivy.mae_loss. This method simply wraps
        the function, and so the docstring for ivy.mae_loss also applies to this method
        with minimal changes.

        Parameters
        ----------
        pred
            input array or container containing predicted labels.
        axis
            the axis along which to compute the mean absolute error. If axis is ``-1``,
            the mean absolute error will be computed along the last dimension.
            Default: ``-1``.
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
            The mean absolute error between the true and predicted values.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> x.mae_loss(y)
        {
            a: ivy.array(1.),
            b: ivy.array(1.)
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.Container(a=ivy.array([2, 3, 4]), b=ivy.array([5, 6, 7]))
        >>> x.mae_loss(y)
        {
            a: ivy.array(1.),
            b: ivy.array(2.)
        }
        """
        return _ContainerWithLossesExperimental._static_mae_loss(
            pred,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
