# global
from typing import Optional, Union, List, Dict

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
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "random_uniform",
            low,
            high,
            shape,
            device=device,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_uniform(
        self: ivy.Container,
        high: Union[float, ivy.Container] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_random_uniform(
            self,
            high,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            dtype=dtype,
            out=out,
        )

    # randint
    @staticmethod
    def static_randint(
        low: Union[int, ivy.Container] = 0.0,
        high: Union[int, ivy.Container] = 1.0,
        shape: Optional[Union[ivy.Shape, ivy.NativeShape, ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.randint. This method simply wraps the
        function, and so the docstring for ivy.randint also applies to this method
        with minimal changes.

        Parameters
        ----------
        low
            Lowest integer that can be drawn from the distribution.
        high
            One above the highest integer that can be drawn from the distribution.
        shape
            a Sequence defining the shape of the output array.
        device
            device on which to create the array. 'cuda:0',
            'cuda:1', 'cpu' etc. (Default value = None).
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Returns an array with the given shape filled with integers from
            the uniform distribution in the “half-open” interval [low, high)

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10), high=20, shape=2)
        >>> print(x)
        {
            a: ivy.array([10, 15]),
            b: ivy.array([16, 12])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=4), high=15, shape=(3,2))
        >>> print(x)
        {
            a: ivy.array([[12, 3],
                         [5, 7],
                         [7, 2]]),
            b: ivy.array([[8, 10],
                         [9, 6],
                         [6, 7]])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=5,b=20,c=40),\
                                      high=100,\
                                      shape=3,\
                                      device='gpu:1')
        >>> print(x)
        {
            a: ivy.array([90, 87, 62]),
            b: ivy.array([52, 95, 37]),
            c: ivy.array([95, 90, 42])
        }

        >>> x = ivy.Container(a=1,b=2)
        >>> y = ivy.Container.randint(low=ivy.Container(a=3,b=5,c=10,d=7),\
                                      high=14,\
                                      shape=5,\
                                      out=x)
        >>> print(x)
        {
            a: ivy.array([4, 10, 13, 3, 3]),
            b: ivy.array([12, 11, 11, 12, 5]),
            c: ivy.array([10, 13, 11, 13, 12]),
            d: ivy.array([12, 7, 8, 11, 8])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10),\
                                      high=ivy.Container(a=5, b= 15, c=2),\
                                      shape=2)
        >>> print(x)
        {
            a: ivy.array([1, 2]),
            b: ivy.array([14, 10])
        }
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
