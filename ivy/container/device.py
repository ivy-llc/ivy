# local
from typing import Union, Literal, Optional, List, Dict

import ivy

# from ivy import DevDistItem
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# Placeholder for type hints.
class DevDistItem:
    pass


# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev_unify_array(
        xs: DevDistItem,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        device: Union[ivy.Device, ivy.NativeDevice],
        mode: Literal["concat", "mean", "sum"],
        axis: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.dev_unify_array. This method
        simply wraps the function, and so the docstring for ivy.dev_unify_array
        also applies to this method with minimal changes.

        Parameters
        ----------
        xs
            The list of arrays to unify onto the specified device.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        device
            The device to unify the arrays to.
        mode
            The mode by which to unify, must be one of [ concat | mean | sum ]
        axis
            The axis along which to concatenate the array, if concat mode is set.
            Default is 0.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the contents of each member of xs, joined as
            specified, each stored on a singular device.

        Examples
        --------

        >>> x_cpu = [0., 0., 0.,]
        >>> x_gpu = [1., 1., 1.,]
        >>> x = {"cpu": ivy.asarray(x_cpu),\
                 "gpu": ivy.asarray(x_gpu, device="gpu:0")}
        >>> y_cpu = [2., 2., 2.,]
        >>> y_gpu = [3., 3., 3.,]
        >>> y = {"cpu": ivy.asarray(y_cpu),\
                 "gpu": ivy.asarray(y_gpu, device="gpu:0")}
        >>> z = ivy.Container(x=ivy.DevDistItem(x), y=ivy.DevDistItem(y))
        >>> z_unified = ivy.dev_unify_array(\
                z, device="cpu", mode="concat", axis=0)\
            )
        >>> print(z_unified)
        {
            x: ivy.array([0., 0., 0., 1., 1., 1.,])
            y: ivy.array([2., 2., 2., 3., 3., 3.,])
        }

        """

        return ContainerBase.multi_map_in_static_method(
            "dev_unify_array",
            xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            device=device,
            mode=mode,
            axis=axis,
        )
