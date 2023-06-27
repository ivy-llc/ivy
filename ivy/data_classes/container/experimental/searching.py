# global
from typing import Optional, Union, List, Dict, Tuple

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithSearchingExperimental(ContainerBase):
    @staticmethod
    def static_unravel_index(
        indices: ivy.Container,
        shape: Tuple[int],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.unravel_index. This method simply
        wraps the function, and so the docstring for ivy.unravel_index also applies to
        this method with minimal changes.

        Parameters
        ----------
        indices
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> indices = ivy.Container(a=ivy.array([22, 41, 37])), b=ivy.array([30, 2]))
        >>> ivy.Container.static_unravel_index(indices, (7,6))
        {
            a: (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
            b: (ivy.array([5, 0], ivy.array([0, 2])))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "unravel_index",
            indices,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def unravel_index(
        self: ivy.Container,
        shape: Tuple[int],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.unravel_index. This method simply
        wraps the function, and so the docstring for ivy.unravel_index also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> indices = ivy.Container(a=ivy.array([22, 41, 37])), b=ivy.array([30, 2]))
        >>> indices.unravel_index((7, 6))
        {
            a: (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
            b: (ivy.array([5, 0], ivy.array([0, 2])))
        }
        """
        return self.static_unravel_index(self, shape, out=out)
