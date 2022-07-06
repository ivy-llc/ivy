# global
from typing import Optional, Union, List, Tuple, Dict
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase
import numpy as np

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithCreation(ContainerBase):
    @staticmethod
    def static_arange(
        start: Number,
        stop: Optional[Number] = None,
        step: Number = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "arange",
            start,
            stop,
            step,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def arange(
        self: ivy.Container,
        x_min: Optional[Union[Number, Union[ivy.Array, ivy.NativeArray]]] = None,
        x_max: Optional[Union[Number, Union[ivy.Array, ivy.NativeArray]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_arange(
            self,
            x_min,
            x_max,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_asarray(
        x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "asarray",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    def asarray(
        self: ivy.Container,
        x: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number], np.ndarray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_asarray(
            self,
            x,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_zeros(
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "zeros",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def zeros(
        self: ivy.Container,
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_zeros(
            self,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_ones(
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "ones",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def ones(
        self: ivy.Container,
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_ones(
            self,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def full_like(
        self: ivy.Container,
        fill_value: Union[int, float],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.full_like(x_, fill_value=fill_value)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            device,
            dtype,
            out,
        )

    def ones_like(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_: ivy.ones_like(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            device,
            dtype,
            out,
        )

    def zeros_like(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_: ivy.zeros_like(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            device,
            dtype,
            out,
        )

    def tril(
        self: ivy.Container,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.tril(x_, k=k) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    def triu(
        self: ivy.Container,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.triu(x_, k=k) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )

    @staticmethod
    def static_empty(
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "empty",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def empty(
        self: ivy.Container,
        shape: Union[int, Tuple[int], List[int]],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_empty(
            self,
            shape,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def empty_like(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_: ivy.empty_like(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            device,
            dtype,
            out,
        )

    @staticmethod
    def static_eye(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "eye",
            n_rows,
            n_cols,
            k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def eye(
        self: ivy.Container,
        n_rows: int,
        n_cols: Optional[int] = None,
        k: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_eye(
            self,
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_linspace(
        start: Union[ivy.Array, ivy.NativeArray, int, float],
        stop: Union[ivy.Array, ivy.NativeArray, int, float],
        num: int,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "linspace",
            start,
            stop,
            num,
            axis,
            endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def linspace(
        self: ivy.Container,
        start: Union[ivy.Array, ivy.NativeArray, int, float],
        stop: Union[ivy.Array, ivy.NativeArray, int, float],
        num: int,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self.static_linspace(
            self,
            start,
            stop,
            num,
            axis,
            endpoint,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
        )

    # TO DO: add meshgrid

    @staticmethod
    def static_full(
        shape: Union[int, Tuple[int, ...]],
        fill_value: Union[int, float],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "full",
            shape,
            fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def full(
        self: ivy.Container,
        shape: Union[int, Tuple[int, ...]],
        fill_value: Union[int, float],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_full(
            self,
            shape,
            fill_value,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def from_dlpack(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_: ivy.from_dlpack(x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
        )

    @staticmethod
    def static_logspace(
        start: Union[ivy.Array, ivy.NativeArray, int, float],
        stop: Union[ivy.Array, ivy.NativeArray, int, float],
        num: int,
        base: float = 10.0,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "logspace",
            start,
            stop,
            num,
            base,
            axis,
            endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            device=device,
        )

    def logspace(
        self: ivy.Container,
        start: Union[ivy.Array, ivy.NativeArray, int, float],
        stop: Union[ivy.Array, ivy.NativeArray, int, float],
        num: int,
        base: float = 10.0,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self.static_logspace(
            self,
            start,
            stop,
            num,
            base,
            axis,
            endpoint,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
        )
