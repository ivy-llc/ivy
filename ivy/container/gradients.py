from typing import Optional, Union, List, Dict, Callable

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithGradients(ContainerBase):
    
    @staticmethod
    def static_is_variable(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        exclusive: bool=False,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "is_variable",
            x,
            exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences  
        )
        
    def is_variable(
        self: ivy.Container,
        exclusive: bool=False,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False
        ):
        return self.static_is_variable(
            self,
            exclusive,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
            )
    
    @staticmethod
    def static_variable_data(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "variable_data",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences  
        )
        
    def variable_data(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
        ):
        return self.static_variable_data(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
            )
        
    @staticmethod
    def static_execute_with_gradients(
        func: Callable,
        xs: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        retain_grads: Optional[bool]=False,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "execute_with_gradients",
            func,
            xs,
            retain_grads,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )
    
    def execute_with_gradients(
        self: ivy.Container,
        xs: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        retain_grads: Optional[bool]=False,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_execute_with_gradients(
            self,
            xs,
            retain_grads,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences
        )
    
    @staticmethod
    def static_adam_step(
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "adam_step",
            dcdw,
            mw,
            vw,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adam_step(
        self: ivy.Container,
        mw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ):
        return self.static_adam_step(
            self,
            mw,
            vw,
            step,
            beta1,
            beta2,
            epsilon,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )


    @staticmethod
    def static_optimizer_update(
        w: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        effective_grad: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "optimizer_update",
            w,
            effective_grad,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def optimizer_update(
        self: ivy.Container,
        effective_grad: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:
        return self.static_optimizer_update(
            self,
            effective_grad,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_gradient_descent_update(
        w: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "gradient_descent_update",
            w,
            dcdw,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def gradient_descent_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ):
        return self.static_gradient_descent_update(
            self,
            dcdw,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lars_update(
        w: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        decay_lambda: Optional[float]=0,
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lars_update",
            w,
            dcdw,
            lr,
            decay_lambda=decay_lambda,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lars_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        decay_lambda: Optional[float]=0,
        inplace: Optional[float]=None,
        stop_gradients: Optional[float]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ):
        return self.static_lars_update(
            self,
            dcdw,
            lr,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_adam_update(
        w: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "adam_update",
            w,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adam_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ):
        return self.static_adam_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lamb_update(
        w: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        max_trust_ratio: Union[int, float]=10,
        decay_lambda: Union[int, float]=0,
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lamb_update",
            w,
            dcdw,
            lr,
            mw_tm1=mw_tm1,
            vw_tm1=vw_tm1,
            step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_trust_ratio=max_trust_ratio,
            decay_lambda=0,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lamb_update(
        self: ivy.Container,
        dcdw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        lr: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        mw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw_tm1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1: Optional[float]=0.9,
        beta2: Optional[float]=0.999,
        epsilon: Optional[float]=1e-7,
        max_trust_ratio: Union[int, float]=10,
        decay_lambda: Union[int, float]=0,
        inplace: Optional[bool]=None,
        stop_gradients: Optional[bool]=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]]=None,
        to_apply: bool=True,
        prune_unapplied: bool=False,
        map_sequences: bool=False,
    ):
        return self.static_lamb_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            max_trust_ratio,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )
