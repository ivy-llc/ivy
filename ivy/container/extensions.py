# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithElementwise(ContainerBase):
    @staticmethod
    def static_sinc(
            x: ivy.Container,
            /,
            *,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),\
                              b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = ivy.Container.static_sinc(x)
        >>> print(y)
        {
            a: ivy.array([0.636, -0.212, 0.127]),
            b: ivy.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),\
                              b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: ivy.array([0.637,-0.212,0.127]),
            b: ivy.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
