from ivy.container.base import ContainerBase
from typing import Union, List, Dict, Optional
import ivy


class ContainerWithNormsExperimental(ContainerBase):

    @staticmethod
    def static_l2_normalize(
            x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
            axis: int = None,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            out=None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.l2_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.l2_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = ivy.Container.static_l2_normalize(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.16903085, 0.50709254, 0.84515423],
                          [0.44183609, 0.56807494, 0.69431382]]),
            b: ivy.array([[-0.70710677, -0.70710677],
                          [-0.89442718, -0.44721359]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "l2_normalize",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l2_normalize(
            self,
            axis=None,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            out=None
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l2_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.l2_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = x.static_l2_normalize(axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.16903085, 0.50709254, 0.84515423],
                          [0.44183609, 0.56807494, 0.69431382]]),
            b: ivy.array([[-0.70710677, -0.70710677],
                          [-0.89442718, -0.44721359]])
        }
        """
        return self.static_l2_normalize(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out
        )
