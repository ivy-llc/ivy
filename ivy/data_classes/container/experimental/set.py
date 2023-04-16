# global
from typing import Union, Optional, List, Dict

# local
from ivy.data_classes.container.base import ContainerBase
import ivy


class _ContainerWithSetExperimental(ContainerBase):
    @staticmethod
    def static_difference(
        x1: ivy.Container,
        x2: ivy.Container = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.difference. This method simply wraps
        the function, and so the docstring for ivy.difference also applies to this
        method with minimal changes.
        Parameters
        ----------
        x1
            a 1D or 2D input container, with a numeric data type.
        x2
            optional second 1D or 2D input array, nativearray, or container, with a
            numeric data type. Must have the same shape as ``self``.
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
        Returns
        -------
        ret
            a container containing the set difference between two containers.
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3., 5.]), b=ivy.array([5., 2., 6., 7., 8.]))
        >>> y = ivy.Container(a=ivy.array([3., 45., 3., 4.]), b=ivy.array([5., 1., 4., 7., 9.]))
        >>> z = ivy.difference(x, y)
        >>> print(z)
        {
            a: ivy.container([1., 2.]),
            b: ivy.container([2., 6., 8.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "difference",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def difference(
        self,
        x2: ivy.Container = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.difference. This method simply wraps
        the function, and so the docstring for ivy.difference also applies to this
        method with minimal changes.
        Parameters
        ----------
        self
            a 1D or 2D input container, with a numeric data type.
        x2
            optional second 1D or 2D input array, nativearray, or container, with a
            numeric data type. Must have the same shape as ``self``.
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
        Returns
        -------
        ret
            a container containing the set difference between two containers.
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3., 5.]), b=ivy.array([5., 2., 6., 7., 8.]))
        >>> y = ivy.Container(a=ivy.array([3., 45., 3., 4.]), b=ivy.array([5., 1., 4., 7., 9.]))
        >>> z = x.difference(y)
        >>> print(z)
        {
            a: ivy.container([1., 2.]),
            b: ivy.container([2., 6., 8.])
        }
        """
        return self.static_difference(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
