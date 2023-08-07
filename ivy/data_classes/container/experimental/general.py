# global
from typing import Optional, Union, List, Dict, Callable, Sequence

# local
from ivy.data_classes.container.base import ContainerBase
import ivy


class _ContainerWithGeneralExperimental(ContainerBase):
    @staticmethod
    def _static_reduce(
        operand: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        init_value: Union[int, float, ivy.Container],
        computation: Union[Callable, ivy.Container],
        /,
        *,
        axes: Union[int, Sequence[int], ivy.Container] = 0,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.reduce. This method simply wraps the
        function, and so the docstring for ivy.reduce also applies to this method with
        minimal changes.

        Parameters
        ----------
        operand
            The array to act on.
        init_value
            The value with which to start the reduction.
        computation
            The reduction function.
        axes
            The dimensions along which the reduction is performed.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one.
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
            The reduced array.

        Examples
        --------
        >>> x = ivy.Container(
        >>>     a=ivy.array([[1, 2, 3], [4, 5, 6]]),
        >>>     b=ivy.native_array([[7, 8, 9], [10, 5, 1]])
        >>> )
        >>> y = ivy.Container.static_reduce(x, 0, ivy.add)
        >>> print(y)
        {
            a: ivy.array([6, 15]),
            b: ivy.array([24, 16])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "reduce",
            operand,
            init_value,
            computation,
            axes=axes,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def reduce(
        self: ivy.Container,
        init_value: Union[int, float, ivy.Container],
        computation: Union[Callable, ivy.Container],
        /,
        *,
        axes: Union[int, Sequence[int], ivy.Container] = 0,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.reduce. This method simply wraps
        the function, and so the docstring for ivy.reduce also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The array to act on.
        init_value
            The value with which to start the reduction.
        computation
            The reduction function.
        axes
            The dimensions along which the reduction is performed.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one.
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
            The reduced array.

        Examples
        --------
        >>> x = ivy.Container(
        >>>     a=ivy.array([[1, 2, 3], [4, 5, 6]]),
        >>>     b=ivy.native_array([[7, 8, 9], [10, 5, 1]])
        >>> )
        >>> y = x.reduce(0, ivy.add)
        >>> print(y)
        {
            a: ivy.array([6, 15]),
            b: ivy.array([24, 16])
        }
        """
        return self._static_reduce(
            self,
            init_value,
            computation,
            axes=axes,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
