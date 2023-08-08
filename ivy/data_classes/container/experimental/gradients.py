from typing import Optional, Union

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithGradientsExperimental(ContainerBase):
    def adagrad_step(
        self: ivy.Container,
        vt: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        epsilon: float = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ivy.adagrad_step(self, vt, epsilon=epsilon, out=out)

    def adagrad_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        vt_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: int,
        /,
        *,
        epsilon: float = 1e-7,
        lr_decay: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ivy.adagrad_update(
            self,
            dcdw,
            lr,
            vt_tm1,
            step,
            epsilon=epsilon,
            lr_decay=lr_decay,
            stop_gradients=stop_gradients,
            out=out,
        )
