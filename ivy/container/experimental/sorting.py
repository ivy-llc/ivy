# global
from typing import Optional, List, Union, Dict

# local
from ivy.container.base import ContainerBase
import ivy


class ContainerWithSortingExperimental(ContainerBase):
    @staticmethod
    def static_msort(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.msort. This method simply wraps the
        function, and so the docstring for ivy.msort also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            array-like or container input.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing sorted input arrays.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> a = ivy.Container(x = ivy.randint(10, size=(2,3)),
        ...                   y = ivy.randint(5, size=(2,2))
        >>> ivy.Container.static_msort(a)
        {
            x: ivy.array(
                [[6, 2, 6],
                 [8, 9, 6]]
                ),
            y: ivy.array(
                [[0, 0],
                 [4, 0]]
                )
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "msort",
            a,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def msort(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.msort.
        This method simply wraps the function, and
        so the docstring for ivy.msort also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container with array-like inputs to sort.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the sorted input arrays.

        Examples
        --------
        >>> a = ivy.Container(x = ivy.randint(10, size=(2,3)),
        ...                   y = ivy.randint(5, size=(2,2))
        >>> a.msort()
        {
            x: ivy.array(
                [[6, 2, 6],
                 [8, 9, 6]]
                ),
            y: ivy.array(
                [[0, 0],
                 [4, 0]]
                )
        }
        """
        return self.static_msort(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
