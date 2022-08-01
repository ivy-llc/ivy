# global
from typing import Optional, List, Union, Dict

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSorting(ContainerBase):
    @staticmethod
    def static_argsort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "argsort",
            x,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def argsort(
        self: ivy.Container,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return self.static_argsort(
            self,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sort(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.add also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]),\
                              b=ivy.array([[8, 1], [5, 0.8]]))
        >>> y = ivy.Container.static_sort(x)
        >>> print(y)
        {
            a: ivy.array([0.2, 5., 9.]),
            b: ivy.array([[1., 8.], [0.8, 5.]])
        }

        >>> x = ivy.Container(a=ivy.array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.Container.static_sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.], [0., 0.4]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sort",
            x,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sort(
        self: ivy.Container,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.sort also applies to this method
        with minimal changes.

        Examples
        --------
        With：code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]),\
                              b=ivy.array([8, 1]))
        >>> y = x.sort()
        >>> print(y)
        {
            a: ivy.array([0.2, 5., 9.]),
            b: ivy.array([1, 8])
        }

        >>> x = ivy.Container(a=ivy.array([5, 9, 0.2]), \
                              b=ivy.array([[8, 1], [5, 0.8]]))
        >>> y = x.sort()
        >>> print(y)
        { a: ivy.array([0.2, 5., 9.]), \
          b: ivy.array([[1., 8.],[0.8, 5.]])
        }

        With：code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.],\
                            [0., 0.4]])
        }

        >>> x = ivy.Container(a=ivy.native_array([8, 0.5, 6]),\
                              b=ivy.array([[9, 0.7], [0.4, 0]]))
        >>> y = ivy.sort(x)
        >>> print(y)
        {
            a: ivy.array([0.5, 6., 8.]),
            b: ivy.array([[0.7, 9.],
                  [0., 0.4]])
        }

        """
        return self.static_sort(
            self,
            axis,
            descending,
            stable,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
    
    @staticmethod
    def static_searchsorted(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        side="left",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "searchsorted",
            x1,
            x2,
            side,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
    
    def searchsorted(
        self: ivy.Container,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        side="left",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        return self.static_searchsorted(
            self,
            x2,
            side,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
