# global
from typing import Union, Optional, List, Dict

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithSetExperimental(ContainerBase):
    @staticmethod
    def static_intersection(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.intersection. This method simply
        wraps the function, and so the docstring for ivy.intersection also applies to
        this method with minimal changes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.array([1., 2., 6.])
        >>> y = ivy.Container(a=ivy.array([3. ,2. ,1., .9]),
        ...                   b=ivy.array([1., 2., 3., 6.]))
        >>> z = ivy.Container.static_intersection(x, y)
        >>> print(z)
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([1., 2., 6.])
        }
        With multiple :class:`ivy.Container` inputs:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3, 9]),
        ...                   b=ivy.array([1, 2, 3, 6]))
        >>> y = ivy.Container(a=ivy.array([4, 2, 1, 8]),
        ...                   b=ivy.array([3, 5, 1, 4]))
        >>> z = ivy.Container.static_intersection(x, y)
        >>> print(z)
        {
            a: ivy.array([1, 2]),
            b: ivy.array([1, 3])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "intersection",
            x1,
            x2,
            assume_unique=assume_unique,
            return_indices=return_indices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def intersection(
        self: ivy.Container,
        x2,
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.intersection. This method simply
        wraps the function, and so the docstring for ivy.intersection also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3, 5]),
        ...                   b=ivy.array([4, 9, 1, 3]))
        >>> y = ivy.Container(a=ivy.array([6, 0, 1, 5]),
        ...                   b=ivy.array([1, 2, 6, 9]))
        >>> z = x.intersection(y)
        >>> print(z)
        {
            a: ivy.array([1, 5]),
            b: ivy.array([1, 9]),
        }
        """
        return self.static_intersection(
            self,
            x2,
            assume_unique=assume_unique,
            return_indices=return_indices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
