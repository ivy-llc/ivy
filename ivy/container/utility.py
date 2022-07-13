# global
from typing import Optional, Union, Dict, Sequence

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithUtility(ContainerBase):
    @staticmethod
    def static_all(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.all. This method simply wraps the
        function, and so the docstring for ivy.all also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = ivy.Container.static_all(x)
        >>> print(y)
        {
            a: ivy.array(False),
            b: ivy.array(False)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "all",
            x,
            axis,
            keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def all(
        self: ivy.Container,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.all. This method simply wraps the
        function, and so the docstring for ivy.all also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = x.all()
        >>> print(y)
        {
            a: ivy.array(False),
            b: ivy.array(False)
        }
        """
        return self.static_all(
            self,
            axis,
            keepdims,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_any(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.any. This method simply wraps the
        function, and so the docstring for ivy.any also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = ivy.Container.static_any(x)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "any",
            x,
            axis,
            keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def any(
        self: ivy.Container,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[Sequence[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.any. This method simply wraps the
        function, and so the docstring for ivy.any also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([0, 1, 1]))
        >>> y = x.any()
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return self.static_any(
            self,
            axis,
            keepdims,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
