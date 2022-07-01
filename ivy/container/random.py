# global
from typing import Optional, Union, List, Dict, Tuple

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithRandom(ContainerBase):
    @staticmethod
    def static_random_uniform(
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "random_uniform",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_uniform(
        self: ivy.Container,
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_random_uniform(
            low,
            high,
            self,
            device,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )


    # randint
    @staticmethod
    def static_randint(
        low: Union[int, ivy.Container] = 0.0,
        high: Union[int, ivy.Container] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.randint. This method simply wraps the
        function, and so the docstring for ivy.randint also applies to this method
        with minimal changes.

        Examples
        --------
        With no device argument and no out argument specified and int arguemnt for shape:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1), high=ivy.Container(b=10), shape=2)
        >>> print(x)
        ivy.array([5,7])

        With no device argument and no out argument specified and sequence for shape:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1), high=ivy.Container(b=10), shape=(3,2))
        >>> print(x)
        ivy.array([[5,8],
                  [9,1],
                  [2,3]])

        With device argument and no out argument specified and int for shape:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1), high=ivy.Container(b=10), shape=3, device='gpu:1')
        >>> print(x)
        ivy.array([4,7,1])

        With no device argument and out argument specified and int for shape:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1), high=ivy.Container(b=10), shape=5, out=x)
        >>> print(x)
        ivy.array([4,7,1,8,5])
        """
        return ContainerBase.multi_map_in_static_method(
            "randint",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            out=out,
        )
