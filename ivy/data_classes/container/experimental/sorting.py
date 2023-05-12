# global
from typing import Optional, List, Union, Dict

# local
from ivy.data_classes.container.base import ContainerBase
import ivy


class _ContainerWithSortingExperimental(ContainerBase):
    @staticmethod
    def static_invert_permutation(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.invert_permutation.

        This method simply wraps the function, and so the docstring for
        ivy.invert_permutation also applies to this method with minimal
        changes.
        """
        return ContainerBase.cont_multi_map_in_function(
            "invert_permutation",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def invert_permutation(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.invert_permutation.

        This method simply wraps the function, and so the docstring for
        ivy.invert_permutation also applies to this method with minimal
        changes.
        """
        return self.static_invert_permutation(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_lexsort(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lexsort. This method simply wraps the
        function, and so the docstring for ivy.lexsort also applies to this method with
        minimal changes.

        Parameters
        ----------
        a
            array-like or container input to sort as keys.
        axis
            axis of each key to be indirectly sorted.
            By default, sort over the last axis of each key.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing sorted input arrays.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> a = ivy.Container(x = ivy.asarray([[9,4,0,4,0,2,1],[1,5,1,4,3,4,4]]),
        ...                   y = ivy.asarray([[1, 5, 2],[3, 4, 4]])
        >>> ivy.Container.static_lexsort(a)
        {
            x: ivy.array([2, 0, 4, 6, 5, 3, 1])),
            y: ivy.array([0, 2, 1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lexsort",
            a,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lexsort(
        self: ivy.Container,
        /,
        *,
        axis: int = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.lexsort. This method simply wraps
        the function, and so the docstring for ivy.lexsort also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container with array-like inputs to sort as keys.
        axis
            axis of each key to be indirectly sorted.
            By default, sort over the last axis of each key.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the sorted input arrays.

        Examples
        --------
        >>> a = ivy.Container(x = ivy.asarray([[9,4,0,4,0,2,1],[1,5,1,4,3,4,4]]),
        ...                   y = ivy.asarray([[1, 5, 2],[3, 4, 4]])
        >>> a.lexsort()
        {
            x: ivy.array([2, 0, 4, 6, 5, 3, 1])),
            y: ivy.array([0, 2, 1])
        }
        """
        return self.static_lexsort(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
