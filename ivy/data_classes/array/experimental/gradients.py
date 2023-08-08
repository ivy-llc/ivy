# global
import abc
from typing import Union, Optional

# local
import ivy


class _ArrayWithGradientsExperimental(abc.ABC):
    def adagrad_step(
        self: ivy.Array,
        vt: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        epsilon: float = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.adagrad_step(self, vt, epsilon=epsilon, out=out)

    def adagrad_update(
        self: ivy.Array,
        dcdw: Union[ivy.Array, ivy.NativeArray],
        lr: Union[float, ivy.Array, ivy.NativeArray],
        vt_tm1: Union[ivy.Array, ivy.NativeArray],
        step: int,
        /,
        *,
        epsilon: float = 1e-7,
        lr_decay: float = 0,
        stop_gradients: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
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
