# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithDataTypes(ContainerBase):
    @staticmethod
    def static_can_cast(
        from_: ivy.Container,
        to: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` static method variant of `ivy.can_cast`. This method simply
        wraps the function, and so the docstring for `ivy.can_cast` also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
            b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(ivy.Container.static_can_cast(x, 'int64'))
        {
            a: false,
            b: true
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "can_cast",
            from_,
            to,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def can_cast(
        self: ivy.Container,
        to: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` instance method variant of `ivy.can_cast`. This method simply
        wraps the function, and so the docstring for `ivy.can_cast` also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
            b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(x.can_cast('int64'))
        {
            a: false,
            b: true
        }
        """
        return self.static_can_cast(
            self, to, key_chains, to_apply, prune_unapplied, map_sequences
        )
